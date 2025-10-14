from collections import defaultdict
import math
import re
import io
import uuid
import torch
import numpy as np
from copy import deepcopy
from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.vision_utils import process_image
from verl.utils.torch_functional import pad_2d_list_to_length
from vlnce_server.client import BatchEnvClient
from vlnce_server.env_config import VLNCEEnvConfig
from dataclasses import asdict
from types import SimpleNamespace
 
def _strip_system_block(text: str) -> str:
    """
    删除 text 中第一个 <|im_start|>system ... <|im_end|> 区块（含标签），
    并返回删除后的字符串。
    如果找不到匹配的开始或结束标签，则返回原文。
    """
    # 非贪婪匹配，匹配跨行
    pattern = r"<\|im_start\|>system.*?<\|im_end\|>"
    # 替换为空
    result = re.sub(pattern, "", text, flags=re.S)
    return result


def _concat_vllm_input(prompt_token_ids, response_token_ids, tokenizer=None):
    # NOTE: temporarily fix qwen-base oov issue
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        output_tensor = torch.cat([
            prompt_token_ids,
            response_token_ids.to(prompt_token_ids.device),
        ], dim=-1)
        return output_tensor.cpu().numpy().flatten().tolist()
    else:
        output_array = np.concatenate([
            prompt_token_ids,
            response_token_ids.cpu().numpy(),
        ], axis=-1)
        return output_array.flatten().tolist()


def _merge_multi_modal_inputs(mm_input, other):
    if not mm_input and not other:
        return {}
    elif len(mm_input) == 0 and len(other) > 0:
        return other
    elif len(mm_input) > 0 and len(other) == 0:
        return mm_input

    output_dict = {}
    for key in mm_input.keys():
        if key not in other.keys():
            output_dict[key] = mm_input[key]
            continue

        mm_value = mm_input[key]
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged_value = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged_value = torch.cat([mm_value, other_value], dim=0)
        else:
            raise ValueError(f"Invalid {type(mm_value)=}, {type(other_value)=}")

        output_dict[key] = merged_value
    return dict(**output_dict, **other)


def _preprocess_multi_modal_inputs(prompt_str, processor, **kwargs):
    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, prompt_str, {}

    vllm_input_prompt = prompt_str.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})
    max_pixels = kwargs.get("max_pixels", 76800)
    min_pixels = kwargs.get("min_pixels", 1024)
    
    image_info_list = []
    for img in input_mm_data["image"]:
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        buf.close()
        img_info = {"bytes": png_bytes, "max_pixels": max_pixels, "min_pixels": min_pixels}
        image_info_list.append(img_info)

    input_mm_data["image"] = [process_image(img) for img in image_info_list]
    model_inputs = processor(text=[vllm_input_prompt], images=input_mm_data["image"], return_tensors="pt")
    input_ids = model_inputs.pop("input_ids")[0]
    attention_mask = model_inputs.pop("attention_mask")[0]

    if "second_per_grid_ts" in model_inputs:
        model_inputs.pop("second_per_grid_ts")

    mm_inputs = dict(model_inputs)
    return vllm_input_prompt, input_ids, mm_inputs


def agent_rollout_loop(config, vllm_engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params):
    from vllm.distributed import parallel_state as vllm_ps

    agent_sampling_params = sampling_params.clone()
    agent_sampling_params.detokenize = True
    agent_sampling_params.skip_special_tokens = False
    agent_sampling_params.spaces_between_special_tokens = False
    agent_sampling_params.n = 1
    agent_sampling_params.include_stop_str_in_output = True
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    agent_sampling_params.max_tokens = max_generated_tokens

    # support custom stop specified in dataset, like </search>, ```, etc.
    custom_stop = list(config.agent.custom_stop)
    if custom_stop:
        prev_stop = sampling_params.stop if sampling_params.stop else []
        agent_sampling_params.stop = prev_stop + custom_stop
        print(f' [DEBUG stop] {type(prev_stop)=}, {type(custom_stop)=}, {type(agent_sampling_params.stop)=}')

    # Refer to: https://github.com/vllm-project/vllm/issues/1728
    # and https://github.com/vllm-project/vllm/issues/15976
    # def process_bad_tokens(token_ids, logits, exclude_token_ids=[]):
    #     for token_id in exclude_token_ids:
    #         logits[token_id] = -9999.999
    #     return logits

    # # NOTE: tmp for visual agent!
    # exclude_func = partial(process_bad_tokens, exclude_token_ids=[
    #     151643,    # <|endoftext|>
    #     151644,    # <|im_start|>
    # ])
    # agent_sampling_params.logits_processors = [exclude_func]
    # agent_sampling_params.bad_words = ["<|endoftext|>", "<|im_start|>"]

    tokenizer = hf_tokenizer(config.agent.vl_model_path)
    processor = hf_processor(config.agent.vl_model_path)

    if multi_modal_inputs is not None:
        multi_modal_inputs = multi_modal_inputs.tolist()
    else:
        multi_modal_inputs = [{}] * len(vllm_inputs)

    batch_size = len(vllm_inputs)
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    running_attn_masks = []
    reward_tensor_list = []
    active_mask = []
    mm_input_list = []
    tool_call_cnt_list = []
    task_success_list = []
    info_list = []
    episode_len = [] # equals to the number of gt actions plus the number of generated actions

    env = ParallelVlnceEnv(config.agent, tokenizer, processor)
    observations, rewards, dones, infos = env.reset(prompts, vllm_inputs, n=sampling_params.n)

    # interleaving inputs if sampling_params.n > 1
    for i in range(batch_size):
        for _ in range(sampling_params.n):
            vllm_input_list.append(deepcopy(vllm_inputs[i]))
            prompt_ids = prompts.batch['input_ids'][i, :].clone()
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch['attention_mask'][i, :].clone()
            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)
            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)
            active_mask.append(True)
            mm_input_list.append(deepcopy(multi_modal_inputs[i]))
            tool_call_cnt_list.append(0)
            task_success_list.append(False)
            info_list.append({})
            episode_len.append(len(prompts[i].non_tensor_batch.get('history_actions')))
    
    pg = vllm_ps.get_tp_group()
    max_total_length = config.prompt_length + config.response_length
    
    # handle gt actions
    for idx, obs in enumerate(observations):
        assert 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys()
        obs_token_ids_vllm = obs['prompt_token_ids_vllm']
        obs_token_ids_model = obs['prompt_token_ids_model'].to(running_states[idx].device)

        if len(vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm) >= max_total_length:
            assert False
        if running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
            assert False

        vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
            vllm_input_list[idx]['prompt_token_ids'], 
            obs_token_ids_vllm,
            tokenizer=tokenizer,
        )

        running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])
        obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=reward_tensor_list[idx].device)
        reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)
        if not config.agent.get('unmask_gt_actions', False):
            obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=running_action_masks[idx].device)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
        else:
            running_action_masks[idx] = torch.cat([running_action_masks[idx], obs["response_mask"].to(running_action_masks[idx].device)])
        attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=running_attn_masks[idx].device)
        running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

        mm_data = obs.get('multi_modal_data', {})
        if 'image' in mm_data.keys():
            if 'multi_modal_data' not in vllm_input_list[idx].keys():
                vllm_input_list[idx]['multi_modal_data'] = {"image": []}
            if 'image' not in vllm_input_list[idx]['multi_modal_data'].keys():
                vllm_input_list[idx]['multi_modal_data']['image'] = []
            vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']

        mm_input = obs.get('multi_modal_inputs', {})
        if mm_input:
            mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

        if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
            assert False
   
    for step in range(config.agent.max_turns + 1):
        print(f' [Trajectory Rollout] turn={step}, bz={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}')
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_inputs = [vinput for vinput, is_active in zip(vllm_input_list, active_mask) if is_active]
        actions = vllm_engine.generate(
            prompts=active_vllm_inputs,
            sampling_params=agent_sampling_params,
            use_tqdm=False
        )
        if pg.is_first_rank:
            obs_results = env.step(active_indices, actions)
        else:
            obs_results = None

        obs_results = pg.broadcast_object(obs_results)
        observations, rewards, dones, infos = obs_results


        for idx, obs, act, rew, done, info in zip(active_indices, observations, actions, rewards, dones, infos):
            # process response token ids
            response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=running_states[idx].device)
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                vllm_input_list[idx]['prompt_token_ids'], 
                response_token_ids,
                tokenizer=tokenizer,
            )

            action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=reward_tensor_list[idx].device)
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device)
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask])

            # Ensure the last token is not obs
            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                assert False
            
            if info['task_success']:
                task_success_list[idx] = True
                assert done

            if done:
                active_mask[idx] = False
                info_list[idx] = info
                continue
            
            tool_call_cnt_list[idx] += len(info['actions'])
            episode_len[idx] += len(info['actions'])

            # process obs tokens and images
            if 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys():
                obs_token_ids_vllm = obs['prompt_token_ids_vllm']
                obs_token_ids_model = obs['prompt_token_ids_model'].to(running_states[idx].device)

                if len(vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm) >= max_total_length:
                    assert False
                if running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                    assert False

                vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                    vllm_input_list[idx]['prompt_token_ids'], 
                    obs_token_ids_vllm,
                    tokenizer=tokenizer,
                )

                running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])
                obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=reward_tensor_list[idx].device)
                reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=running_action_masks[idx].device)
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=running_attn_masks[idx].device)
                running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

                mm_data = obs.get('multi_modal_data', {})
                if 'image' in mm_data.keys():
                    if 'multi_modal_data' not in vllm_input_list[idx].keys():
                        vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                    vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']

                mm_input = obs.get('multi_modal_inputs', {})
                if mm_input:
                    mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

            if running_states[idx].shape[-1] >= max_total_length or len(vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                assert False
    
    is_validate = prompts.meta_info.get("validate", False)

    # perform dynamic sampling if no trajectory in the group successfully reaches the goal
    if not is_validate and config.agent.get('enable_dynamic_sampling', False):
        max_sample_attempts = int(config.agent.get('max_sample_attempts'))
        assert max_sample_attempts > 0
        n_attempts = 0
        while n_attempts < max_sample_attempts:
            n_attempts += 1
            group_success_tensor = torch.tensor(task_success_list, dtype=int).reshape(batch_size, sampling_params.n)
            group_success_count = group_success_tensor.sum(dim=1)
            groups_need_resample = torch.where(group_success_count < 1)[0]
            if len(groups_need_resample) <= 0:
                break
            ds_prompts = prompts[groups_need_resample]
            ds_vllm_inputs = [vllm_inputs[i] for i in groups_need_resample]
            ds_multi_modal_inputs = [multi_modal_inputs[i] for i in groups_need_resample]

            ds_observations, ds_rewards, ds_dones, ds_infos = env.reset(ds_prompts, ds_vllm_inputs, n=sampling_params.n)

            ds_vllm_input_list = []
            ds_running_states = []
            ds_running_action_masks = []
            ds_running_attn_masks = []
            ds_reward_tensor_list = []
            ds_active_mask = []
            ds_mm_input_list = []
            ds_tool_call_cnt_list = []
            ds_task_success_list = []
            ds_info_list = []
            ds_episode_len = []
            
            for i in range(len(groups_need_resample)):
                for _ in range(sampling_params.n):
                    ds_vllm_input_list.append(deepcopy(ds_vllm_inputs[i]))
                    prompt_ids = ds_prompts.batch['input_ids'][i, :].clone()
                    ds_running_states.append(prompt_ids)
                    prompt_mask = ds_prompts.batch['attention_mask'][i, :].clone()
                    ds_running_action_masks.append(prompt_mask)
                    ds_running_attn_masks.append(prompt_mask)
                    reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
                    ds_reward_tensor_list.append(reward_tensor)
                    ds_active_mask.append(True)
                    ds_mm_input_list.append(deepcopy(ds_multi_modal_inputs[i]))
                    ds_tool_call_cnt_list.append(0)
                    ds_task_success_list.append(False)
                    ds_info_list.append({})
        
                    n_history_actions = len(ds_prompts[i].non_tensor_batch.get('history_actions'))
                    ds_episode_len.append(n_history_actions)
                
            # handle gt actions
            for idx, obs in enumerate(ds_observations):
                assert 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys()
                obs_token_ids_vllm = obs['prompt_token_ids_vllm']
                obs_token_ids_model = obs['prompt_token_ids_model'].to(ds_running_states[idx].device)

                if len(ds_vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm) >= max_total_length:
                    assert False
                if ds_running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                    assert False

                ds_vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                    ds_vllm_input_list[idx]['prompt_token_ids'], 
                    obs_token_ids_vllm,
                    tokenizer=tokenizer,
                )
                
                ds_running_states[idx] = torch.cat([ds_running_states[idx], obs_token_ids_model])
                obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=ds_reward_tensor_list[idx].device)
                ds_reward_tensor_list[idx] = torch.cat([ds_reward_tensor_list[idx], obs_reward], dim=-1)
               
                if not config.agent.get('unmask_gt_actions', False):
                    obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=ds_running_action_masks[idx].device)
                    ds_running_action_masks[idx] = torch.cat([ds_running_action_masks[idx], obs_mask])
                else:
                    ds_running_action_masks[idx] = torch.cat([ds_running_action_masks[idx], obs["response_mask"].to(ds_running_action_masks[idx].device)])
                
                attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=ds_running_attn_masks[idx].device)
                ds_running_attn_masks[idx] = torch.cat([ds_running_attn_masks[idx], attn_mask])                

                mm_data = obs.get('multi_modal_data', {})
                if 'image' in mm_data.keys():
                    if 'multi_modal_data' not in ds_vllm_input_list[idx].keys():
                        ds_vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                    if 'image' not in ds_vllm_input_list[idx]['multi_modal_data'].keys():
                        ds_vllm_input_list[idx]['multi_modal_data']['image'] = []
                    ds_vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']


                mm_input = obs.get('multi_modal_inputs', {})
                if mm_input:
                    ds_mm_input_list[idx] = _merge_multi_modal_inputs(ds_mm_input_list[idx], mm_input)

                if ds_running_states[idx].shape[-1] >= max_total_length or len(ds_vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                    assert False
                    
            for step in range(config.agent.max_turns + 1):
                print(f' [Trajectory Rollout (Dynamic Sampling #{n_attempts})] turn={step}, bz={len(groups_need_resample)}, n={sampling_params.n}, num_active={sum(ds_active_mask)}')
                if sum(ds_active_mask) == 0:
                    break

                ds_active_indices = [idx for idx, is_active in enumerate(ds_active_mask) if is_active]
                ds_active_vllm_inputs = [vinput for vinput, is_active in zip(ds_vllm_input_list, ds_active_mask) if is_active]
                ds_actions = vllm_engine.generate(
                    prompts=ds_active_vllm_inputs,
                    sampling_params=agent_sampling_params,
                    use_tqdm=False
                )
                if pg.is_first_rank:
                    obs_results = env.step(ds_active_indices, ds_actions)
                else:
                    obs_results = None
                obs_results = pg.broadcast_object(obs_results)
                ds_observations, ds_rewards, ds_dones, ds_infos = obs_results
               
                for idx, obs, act, rew, done, info in zip(ds_active_indices, ds_observations, ds_actions, ds_rewards, ds_dones, ds_infos):
                    # process response token ids
                    response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=ds_running_states[idx].device)
                    ds_running_states[idx] = torch.cat([ds_running_states[idx], response_token_ids])
                    ds_vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                        ds_vllm_input_list[idx]['prompt_token_ids'], 
                        response_token_ids,
                        tokenizer=tokenizer,
                    )

                    action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=ds_reward_tensor_list[idx].device)
                    ds_reward_tensor_list[idx] = torch.cat([ds_reward_tensor_list[idx], action_reward])
                    ds_reward_tensor_list[idx][-1] += rew

                    action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=ds_running_action_masks[idx].device)
                    ds_running_action_masks[idx] = torch.cat([ds_running_action_masks[idx], action_mask])
                    ds_running_attn_masks[idx] = torch.cat([ds_running_attn_masks[idx], action_mask])

                    # Ensure the last token is not obs
                    if ds_running_states[idx].shape[-1] >= max_total_length or len(ds_vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                        assert False
                    
                    if info['task_success']:
                        ds_task_success_list[idx] = True
                        assert done

                    if done:
                        ds_active_mask[idx] = False
                        ds_info_list[idx] = {
                            **info,
                            "from_dynamic_sampling": True,
                        }
                        continue

                    ds_tool_call_cnt_list[idx] += len(info['actions'])
                    ds_episode_len[idx] += len(info['actions'])

                    # process obs tokens and images
                    if 'prompt_token_ids_vllm' in obs.keys() and 'prompt_token_ids_model' in obs.keys():
                        obs_token_ids_vllm = obs['prompt_token_ids_vllm']
                        obs_token_ids_model = obs['prompt_token_ids_model'].to(ds_running_states[idx].device)

                        if len(ds_vllm_input_list[idx]['prompt_token_ids']) + len(obs_token_ids_vllm) >= max_total_length:
                            assert False
                        if ds_running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                            assert False
                        
                        ds_vllm_input_list[idx]['prompt_token_ids'] = _concat_vllm_input(
                            ds_vllm_input_list[idx]['prompt_token_ids'], 
                            obs_token_ids_vllm,
                            tokenizer=tokenizer,
                        )

                        ds_running_states[idx] = torch.cat([ds_running_states[idx], obs_token_ids_model])
                        obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=ds_reward_tensor_list[idx].device)
                        ds_reward_tensor_list[idx] = torch.cat([ds_reward_tensor_list[idx], obs_reward], dim=-1)

                        obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=ds_running_action_masks[idx].device)
                        ds_running_action_masks[idx] = torch.cat([ds_running_action_masks[idx], obs_mask])
                        attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=ds_running_attn_masks[idx].device)
                        ds_running_attn_masks[idx] = torch.cat([ds_running_attn_masks[idx], attn_mask])

                        mm_data = obs.get('multi_modal_data', {})
                        if 'image' in mm_data.keys():
                            if 'multi_modal_data' not in ds_vllm_input_list[idx].keys():
                                ds_vllm_input_list[idx]['multi_modal_data'] = {"image": []}
                            ds_vllm_input_list[idx]['multi_modal_data']['image'] += mm_data['image']

                        
                        mm_input = obs.get('multi_modal_inputs', {})
                        if mm_input:
                            ds_mm_input_list[idx] = _merge_multi_modal_inputs(ds_mm_input_list[idx], mm_input)

                    if ds_running_states[idx].shape[-1] >= max_total_length or len(ds_vllm_input_list[idx]['prompt_token_ids']) >= max_total_length:
                        assert False
                        
            for i, idx in enumerate(groups_need_resample):
                from_range = slice(idx * sampling_params.n, (idx + 1) * sampling_params.n)
                to_range = slice(i * sampling_params.n, (i + 1) * sampling_params.n)

                running_states[from_range] = ds_running_states[to_range]
                running_action_masks[from_range] = ds_running_action_masks[to_range]
                running_attn_masks[from_range] = ds_running_attn_masks[to_range]
                reward_tensor_list[from_range] = ds_reward_tensor_list[to_range]
                tool_call_cnt_list[from_range] = ds_tool_call_cnt_list[to_range]
                task_success_list[from_range] = ds_task_success_list[to_range]
                episode_len[from_range] = ds_episode_len[to_range]
                info_list[from_range] = ds_info_list[to_range]
                mm_input_list[from_range] = ds_mm_input_list[to_range]
    
    # rollout fallback
    if not is_validate and config.agent.get('enable_fallback', False):
        group_success_tensor = torch.tensor(task_success_list, dtype=int).reshape(batch_size, sampling_params.n)
        group_success_count = group_success_tensor.sum(dim=1)
        groups_need_fallback = torch.where(group_success_count < 1)[0]
        if len(groups_need_fallback) > 0:
            
            fb_prompts = prompts[groups_need_fallback]
            fb_vllm_inputs = [vllm_inputs[i] for i in groups_need_fallback]
            fb_multi_modal_inputs = [multi_modal_inputs[i] for i in groups_need_fallback]

            fb_observations, fb_rewards, fb_dones, fb_infos = env.reset(fb_prompts, fb_vllm_inputs, n=1)

            fb_running_states = []
            fb_running_action_masks = []
            fb_running_attn_masks = []
            fb_reward_tensor_list = []
            fb_active_mask = []
            fb_mm_input_list = []
            fb_tool_call_cnt_list = []
            fb_task_success_list = []
            fb_info_list = []
            fb_episode_len = []
            
            fb_actions_list = []

            for i in range(len(groups_need_fallback)):
                prompt_ids = fb_prompts.batch['input_ids'][i, :].clone()
                fb_running_states.append(prompt_ids)
                prompt_mask = fb_prompts.batch['attention_mask'][i, :].clone()
                fb_running_action_masks.append(prompt_mask)
                fb_running_attn_masks.append(prompt_mask)
                reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
                fb_reward_tensor_list.append(reward_tensor)
                fb_active_mask.append(True)
                fb_mm_input_list.append(deepcopy(fb_multi_modal_inputs[i]))
                fb_tool_call_cnt_list.append(0)
                fb_task_success_list.append(False)
                fb_info_list.append({})
                
                n_history_actions = len(fb_prompts[i].non_tensor_batch.get('history_actions'))
                fb_episode_len.append(n_history_actions)
                
                full_gt_actions = fb_prompts[i].non_tensor_batch.get('full_gt_action_seqs')
                fb_actions = full_gt_actions[n_history_actions:]
                fb_actions_list.append(fb_actions)
                assert len(fb_actions) >= 1
        
            max_total_length = config.prompt_length + config.response_length
        
            # handle gt actions
            for idx, obs in enumerate(fb_observations):
                assert 'prompt_token_ids_model' in obs.keys()
                obs_token_ids_model = obs['prompt_token_ids_model'].to(fb_running_states[idx].device)

                if fb_running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                    assert False

                fb_running_states[idx] = torch.cat([fb_running_states[idx], obs_token_ids_model])
                obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=fb_reward_tensor_list[idx].device)
                fb_reward_tensor_list[idx] = torch.cat([fb_reward_tensor_list[idx], obs_reward], dim=-1)
               
                if not config.agent.get('unmask_gt_actions', False):
                    obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=fb_running_action_masks[idx].device)
                    fb_running_action_masks[idx] = torch.cat([fb_running_action_masks[idx], obs_mask])
                else:
                    fb_running_action_masks[idx] = torch.cat([fb_running_action_masks[idx], obs["response_mask"].to(fb_running_action_masks[idx].device)])
                
                attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=fb_running_attn_masks[idx].device)
                fb_running_attn_masks[idx] = torch.cat([fb_running_attn_masks[idx], attn_mask])                

                mm_input = obs.get('multi_modal_inputs', {})
                if mm_input:
                    fb_mm_input_list[idx] = _merge_multi_modal_inputs(fb_mm_input_list[idx], mm_input)

                if fb_running_states[idx].shape[-1] >= max_total_length:
                    assert False
    
    
            max_actions_per_step = config.agent.max_actions_per_step
            action_sep = config.agent.action_sep
            
            fb_max_turns = math.ceil(max(len(fb_actions) for fb_actions in fb_actions_list) / max_actions_per_step)
            assert fb_max_turns <= config.agent.max_turns
            
            for step in range(config.agent.max_turns + 1):
                print(f' [Trajectory Rollout (Fallback)] turn={step}, bz={len(groups_need_fallback)}, n={1}, num_active={sum(fb_active_mask)}')

                if sum(fb_active_mask) == 0:
                    break

                fb_active_indices = [idx for idx, is_active in enumerate(fb_active_mask) if is_active]
               
                # fake vllm generation output                
                left_bound = step * max_actions_per_step
                right_bound = left_bound + max_actions_per_step
                fb_act = [f"{(action_sep + ' ').join(fb_actions_list[idx][left_bound:right_bound])}<|im_end|>" for idx in fb_active_indices]
                
                fb_token_ids = [tokenizer.encode(act, add_special_tokens=False) for act in fb_act]
                fb_actions = [SimpleNamespace(outputs=[SimpleNamespace(token_ids=token_ids, text=act, finish_reason='')]) for act, token_ids in zip(fb_act, fb_token_ids)]
                
                obs_results = env.step(fb_active_indices, fb_actions)
                fb_observations, fb_rewards, fb_dones, fb_infos = obs_results

                for idx, obs, act, rew, done, info in zip(fb_active_indices, fb_observations, fb_actions, fb_rewards, fb_dones, fb_infos):
                    # process response token ids
                    response_token_ids = torch.tensor(act.outputs[0].token_ids, dtype=torch.int64, device=fb_running_states[idx].device)
                    fb_running_states[idx] = torch.cat([fb_running_states[idx], response_token_ids])

                    action_reward = torch.zeros_like(response_token_ids, dtype=torch.float, device=fb_reward_tensor_list[idx].device)
                    fb_reward_tensor_list[idx] = torch.cat([fb_reward_tensor_list[idx], action_reward])
                    fb_reward_tensor_list[idx][-1] += rew

                    action_mask = torch.ones_like(response_token_ids, dtype=torch.int64, device=fb_running_action_masks[idx].device)
                    fb_running_action_masks[idx] = torch.cat([fb_running_action_masks[idx], action_mask])
                    fb_running_attn_masks[idx] = torch.cat([fb_running_attn_masks[idx], action_mask])

                    # Ensure the last token is not obs
                    if fb_running_states[idx].shape[-1] >= max_total_length:
                        assert False
                    
                    if info['task_success']:
                        fb_task_success_list[idx] = True
                        assert done

                    if done:
                        fb_active_mask[idx] = False
                        fb_info_list[idx] = {
                            **info,
                            "is_fallback": True,
                        }
                        continue
                    
                    fb_tool_call_cnt_list[idx] += len(info['actions'])
                    fb_episode_len[idx] += len(info['actions'])

                    # process obs tokens and images
                    if 'prompt_token_ids_model' in obs.keys():
                        obs_token_ids_model = obs['prompt_token_ids_model'].to(fb_running_states[idx].device)

                        if fb_running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                            assert False

                        fb_running_states[idx] = torch.cat([fb_running_states[idx], obs_token_ids_model])
                        obs_reward = torch.zeros(len(obs_token_ids_model), dtype=torch.float, device=fb_reward_tensor_list[idx].device)
                        fb_reward_tensor_list[idx] = torch.cat([fb_reward_tensor_list[idx], obs_reward], dim=-1)

                        obs_mask = torch.zeros(len(obs_token_ids_model), dtype=torch.int64, device=fb_running_action_masks[idx].device)
                        fb_running_action_masks[idx] = torch.cat([fb_running_action_masks[idx], obs_mask])
                        attn_mask = torch.ones(len(obs_token_ids_model), dtype=torch.int64, device=fb_running_attn_masks[idx].device)
                        fb_running_attn_masks[idx] = torch.cat([fb_running_attn_masks[idx], attn_mask])

                        mm_input = obs.get('multi_modal_inputs', {})
                        if mm_input:
                            fb_mm_input_list[idx] = _merge_multi_modal_inputs(fb_mm_input_list[idx], mm_input)

                    if fb_running_states[idx].shape[-1] >= max_total_length:
                        assert False
                                                
            assert all(fb_task_success_list)

            # Restore the fallback results to the original lists.
            # Here, we replace the last item in each group that requires a fallback with the corresponding fallback results.
            for i, idx in enumerate(groups_need_fallback):
                idx_to_replace = idx * sampling_params.n + (sampling_params.n - 1)
                running_states[idx_to_replace] = fb_running_states[i]
                running_action_masks[idx_to_replace] = fb_running_action_masks[i]
                running_attn_masks[idx_to_replace] = fb_running_attn_masks[i]
                reward_tensor_list[idx_to_replace] = fb_reward_tensor_list[i]
                tool_call_cnt_list[idx_to_replace] = fb_tool_call_cnt_list[i]
                task_success_list[idx_to_replace] = fb_task_success_list[i]
                episode_len[idx_to_replace] = fb_episode_len[i]
                info_list[idx_to_replace] = fb_info_list[i]
                mm_input_list[idx_to_replace] = fb_mm_input_list[i]
    
    env.close()
    target_device = prompts.batch['input_ids'].device
    running_states = [state[: max_total_length] for state in running_states]
    state_tensor = pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device)

    running_action_masks = [mask[: max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)

    running_attn_masks = [mask[: max_total_length] for mask in running_attn_masks]
    attn_mask_tensor = pad_2d_list_to_length(running_attn_masks, 0, max_total_length).to(target_device)
    if processor is not None and processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
        # For Qwen-VL: (n*bs, 3, seq_len)
        position_ids_list = [
            get_rope_index(
                processor,
                input_ids=state_tensor[i, :],
                image_grid_thw=mm_input_list[i].get("image_grid_thw", None),
                video_grid_thw=mm_input_list[i].get("video_grid_thw", None),
                second_per_grid_ts=mm_input_list[i].get("second_per_grid_ts", None),
                attention_mask=attn_mask_tensor[i, :],
            ) for i in range(batch_size * sampling_params.n)
        ]
        position_ids_tensor = torch.stack(position_ids_list, dim=0)
    else:
        # For LM: (n*bs, seq_len)
        position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)

    reward_tensor_list = [reward[: max_total_length] for reward in reward_tensor_list]
    reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)

    tool_call_tensor = torch.tensor(tool_call_cnt_list, dtype=torch.float32).to(target_device).unsqueeze(1)
    task_success_tensor = torch.tensor(task_success_list, dtype=torch.bool).to(target_device).unsqueeze(1)
    episode_len_tensor = torch.tensor(episode_len, dtype=torch.int32).to(target_device).unsqueeze(1)
    
    non_tensors_dict = {}
    if processor is not None:
        non_tensors_dict["multi_modal_inputs"] = mm_input_list
    non_tensors_dict["info"] = info_list
    
    return DataProto.from_dict(
        tensors={
            "response": state_tensor[:, -config.response_length: ],
            "action_mask": action_mask_tensor,
            "attention_mask": attn_mask_tensor,
            "position_ids": position_ids_tensor,
            "env_reward": reward_tensor[:, -config.response_length: ],
            "tool_cnt": tool_call_tensor,
            "task_success": task_success_tensor,
            "episode_len": episode_len_tensor,
        },
        # non_tensors={"multi_modal_inputs": mm_input_list} if processor is not None else None
        non_tensors=non_tensors_dict
    )


class ParallelVlnceEnv:
    """
    The interface is designed to be the similar to : https://github.com/openai/gym
    """
    def __init__(self, cfg, tokenizer, processor, **kwargs):
        self.config = cfg
        self.tokenizer = tokenizer
        self.processor = processor
        
        self.env_client = BatchEnvClient(base_url=cfg.base_url ,timeout=cfg.timeout)

        self.index_2_env_id = {} # map index to a unique env id (not env config id)
        self.env_id_2_config = {} # map env id to its env config

    def step(self, active_indices, actions):
        """
        Input:
        - actions: vllm.RequestOutput

        Output:
        - observations: List[Dict], content like {"prompt_token_ids": ..., "multi_modal_data": ...}, 
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to 
                the last valid token of model outputs. This might be useful for ..., 
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        assert len(active_indices) == len(actions), f"{len(active_indices)=}, {len(actions)=}"
        obs_list = [{}] * len(actions)
        reward_list = [0.0] * len(actions) 
        done_list = [False] * len(actions) 
        info_list = [{}] * len(actions)
        
        valid_indices = []
        real_indices = []
        valid_actions = []
        
        # 1. filtering valid actions
        for i, (idx, act) in enumerate(zip(active_indices, actions)):

            if len(act.outputs[0].token_ids) == 0:
                assert False

            real_indices.append(i)
            valid_indices.append(idx)
            
            act_text = act.outputs[0].text
            if act_text.endswith("<|im_end|>"):
                valid_actions.append(act_text[:-len("<|im_end|>")])
            else:
                valid_actions.append(act_text)
                print(f"[WARN] expect {act_text=} ends with <|im_end|>")

        ids2actions = {}

        env_id_2_real_index = {}
        for i, idx, action in zip(real_indices, valid_indices, valid_actions):
            env_id = self.index_2_env_id[idx]
            ids2actions[env_id] = action
            env_id_2_real_index[env_id] = i
        
        step_results = self.env_client.step_batch(ids2actions)

        env_id_2_index = {v: k for k, v in self.index_2_env_id.items()}
        for env_id, rst in step_results.items():
            idx = env_id_2_index[env_id]
            real_idx = env_id_2_real_index[env_id]
            obs, reward, done, info = rst
            if done:
                print(info)
            multi_modal_data = {
                "image": [img for img in obs["multi_modal_data"]['<image>']]
            }
            messages = [
                {"role": "user", "content": obs['obs_str']}
            ]
            prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_str = _strip_system_block(prompt_str)
            prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(prompt_str, self.processor, multi_modal_data=multi_modal_data, max_pixels=self.config.max_pixels, min_pixels=self.config.min_pixels)
            obs_token_ids_vllm = self.tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors='pt')[0]
            tool_result_info = {
                "prompt_token_ids_vllm": obs_token_ids_vllm,
                "prompt_token_ids_model": obs_token_ids_model,
                "multi_modal_inputs": mm_inputs,
                "multi_modal_data": multi_modal_data
            }       
            
            reward_list[real_idx] = reward
            obs_list[real_idx] = tool_result_info
            done_list[real_idx] = done 
            info_list[real_idx] = info
            
        return obs_list, reward_list, done_list, info_list

    def reset(self, prompts, vllm_inputs, n=1, **kwargs):
        assert len(prompts) == len(vllm_inputs), f"{len(prompts)=}, {len(vllm_inputs)=}"

        
        self.index_2_env_id = {}
        env_buckets = defaultdict(set) # maps env_config_id to env_id
        env_ids_reset = []
        env_ids_create = []
        idx = 0
                    
        for env_id, cfg in self.env_id_2_config.items():
            bucket_key = cfg.config_id()
            env_buckets[bucket_key].add(env_id)
            
        
        for i in range(len(prompts)):
            data_item = prompts[i]  # DataProtoItem
            tool_name = data_item.non_tensor_batch.pop(self.config.tool_name_key, '')
            raw_prompt = data_item.non_tensor_batch.pop('raw_prompt', None)
            
            assert tool_name == 'vlnce'
            vllm_input_item = vllm_inputs[i]   # {"prompt_token_ids": ..., "multi_modal_data": ...}
            multi_modal_data = vllm_input_item.get("multi_modal_data", None)
            
            episode_id = data_item.non_tensor_batch.pop('episode_id', None)
            history_actions = data_item.non_tensor_batch.pop('history_actions', None)
            data_source = data_item.non_tensor_batch.pop('data_source', None)
            step_budget = data_item.non_tensor_batch.pop('step_budget', None)
            turn_budget = data_item.non_tensor_batch.pop('turn_budget', None) 
            assert episode_id is not None and history_actions is not None and data_source is not None
            assert step_budget is not None
            
            history_actions = history_actions.tolist() if isinstance(history_actions, (np.ndarray)) else history_actions
            assert isinstance(history_actions, (list, tuple))

            episode_id = int(episode_id)
            data_source = str(data_source)
            step_budget = int(step_budget)
            turn_budget = int(turn_budget)
            
            for _ in range(n):
                env_config = VLNCEEnvConfig(
                    episode_id=episode_id, 
                    history_actions=history_actions, 
                    data_source=data_source,
                    action_space="r2r",

                    step_budget=step_budget,
                    turn_budget=turn_budget,

                    format_reward=0, # no format reward
                    success_reward_base=self.config.reward.success_reward_base,
                    ndtw_reward_base=self.config.reward.ndtw_reward_base,
                    reward_type=self.config.reward.reward_type,

                    max_actions_per_step=self.config.max_actions_per_step,
                    action_sep=self.config.action_sep,
                    prompt_format=self.config.prompt_format,

                    save_as_video=self.config.save_as_video,
                    experiment_name=self.config.experiment_name,
                )
                env_config_id = env_config.config_id()

                # Check if we can reuse the environment that have the same config
                if env_config_id in env_buckets and env_buckets[env_config_id]:
                    
                    old_env_id = env_buckets[env_config_id].pop()
                    env_ids_reset.append(old_env_id)
                    self.index_2_env_id[idx] = old_env_id
                else:
                    
                    env_id = str(uuid.uuid4())
                    env_ids_create.append(env_id)
                    self.env_id_2_config[env_id] = env_config
                    self.index_2_env_id[idx] = env_id

                idx += 1
        
        # Step 2: Collect ids which need to be closed 
        env_ids_to_close=[]
        # Close unused environments
        for bucket_key, env_ids in env_buckets.items():
            for env_id in env_ids:
                env_ids_to_close.append(env_id)
                self.env_id_2_config.pop(env_id)

        # Step 3: Close unused environments
        self.env_client.close_batch(env_ids_to_close)

        # Step 4: Create new environments
        ids2configs_create = {env_id: {
            'env_name': 'vlnce',
            'env_config': asdict(self.env_id_2_config[env_id]),
        } for env_id in env_ids_create}
        env_ids_reset.extend(env_ids_create)
        self.env_client.create_environments_batch(ids2configs_create)


        # Step 5: Reset environments
        assert len(env_ids_reset) == len(prompts) * n 
        # {env_id: seed} Currently seed is not used.
        reset_results=self.env_client.reset_batch({env_id: 42 for env_id in env_ids_reset})            
            
        n_total = n * len(prompts)
        obs_list = []
        for idx in range(n_total):
            env_id = self.index_2_env_id[idx]
            reset_res = reset_results[env_id]
            observations, _ = reset_res
            multi_modal_data = {
                "image": [img for obs in observations for img in obs["multi_modal_data"]['<image>']]
            }
            messages = [
                {"role": "user", "content": observations[0]['obs_str']}
            ]
            for j, obs in enumerate(observations[1:]):
                messages.append({
                    "role": "assistant", "content": obs['response']
                })
                messages.append({
                    "role": "user", "content": obs['obs_str']
                })
            assert len(multi_modal_data['image'])*2-1 == len(messages)
            prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_str = _strip_system_block(prompt_str)
            prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(prompt_str, self.processor, multi_modal_data=multi_modal_data, max_pixels=self.config.max_pixels, min_pixels=self.config.min_pixels)
            obs_token_ids_vllm = self.tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors='pt')[0]
                
            start_tokens = torch.tensor([151644, 77091, 198]) # <|im_start|>assistant\n
            end_tokens = torch.tensor([151645]) # <|im_end|>
            im_start_locs = torch.sort(torch.where(obs_token_ids_model == 151644)[0]).values[:-1] # exclude the last <|im_start|> token ( generation prompt )
            im_end_locs = torch.sort(torch.where(obs_token_ids_model == 151645)[0]).values
            
            response_mask = torch.zeros_like(obs_token_ids_model, dtype=torch.int64)
            
            n_response_blocks = 0
            for start_loc in im_start_locs:
                if torch.equal(obs_token_ids_model[start_loc: start_loc + len(start_tokens)], start_tokens):
                    end_candidates = torch.where(im_end_locs >= start_loc + len(start_tokens))[0]
                    assert len(end_candidates) > 0, f"no end token found after start token at {start_loc}"
                    end_loc = im_end_locs[torch.min(end_candidates)]
                    
                    response_mask[start_loc + len(start_tokens): end_loc + 1] = 1 # include <|im_end|> 
                    n_response_blocks += 1
            
            assert n_response_blocks == len(multi_modal_data['image']) - 1
            
            tool_result_info = {
                "prompt_token_ids_vllm": obs_token_ids_vllm,
                "prompt_token_ids_model": obs_token_ids_model,
                "multi_modal_inputs": mm_inputs,
                "multi_modal_data": multi_modal_data,
                "response_mask": response_mask,
            }
            obs_list.append(tool_result_info)
       
        
        return obs_list, [0.0]*n_total, [False]*n_total, {}

    def close(self):
        ids_to_close = list(self.index_2_env_id.values())
        self.env_client.close_batch(ids_to_close)
