import os
import time
import json
import numpy as np
import re
import cv2
import random
import argparse
from PIL import Image
import multiprocessing as mp
from openai import OpenAI
from tqdm import tqdm
import imageio
import base64
from io import BytesIO
from PIL import Image
import habitat
from habitat import Env
from habitat.utils.visualizations import maps
from VLN_CE.vlnce_baselines.config.default import get_config
from qwen_vl_utils.vision_process import smart_resize

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

SYSTEM_PROMPT_R2R = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)

SYSTEM_PROMPT_RxR = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, turn left 30 degrees, turn left 60 degrees, turn left 90 degrees, turn right 30 degrees, turn right 60 degrees, turn right 90 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)

first_turn_user_prompt = ("Instruction: {}"
        "Decide your next action. "
        "You can take up to 3 actions at a time, separated by ','. ")

normal_user_prompt = ("Instruction: {}"
        "Decide your next action. "
        "You can take up to 3 actions at a time, separated by ','. ")


def evaluate_agent(result_queue, action_space, api_key, base_url, config, dataset, result_path, num_generations,
                    max_pixels, max_turns) -> None:
 
    for i in range(num_generations):

        env = Env(config.TASK_CONFIG, dataset)

        agent = ActiveVlnAgent(action_space, api_key, base_url, result_path, max_pixels, max_turns, num_generations=num_generations)

        num_episodes = len(env.episodes)
        
        EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
        EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

        target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

        count = 0
        
        for _ in range(num_episodes):
                
            episode_start_time = time.time()
            
            obs = env.reset()
            iter_step = 0
            agent.reset()

            t_dict = {
                "t_episode": 0,
            }

            continuse_rotation_count = 0
            last_dtg = 999
            if os.path.exists(os.path.join(os.path.join(result_path, "log"),"stats_{}_{}.json".format(env.current_episode.episode_id, i))):
                t_dict["t_episode"] = time.time() - episode_start_time
                result_queue.put(t_dict)
                continue
                        
            early_stop_reason = ''
            
            while not env.episode_over:
                
                info = env.get_metrics()
                
                if info["distance_to_goal"] != last_dtg:
                    last_dtg = info["distance_to_goal"]
                    continuse_rotation_count=0
                else :
                    continuse_rotation_count +=1 
                
                
                action = agent.act(obs, info, env.current_episode.episode_id)

                if continuse_rotation_count > EARLY_STOP_ROTATION:
                    action = {"action": 0}
                    early_stop_reason = "early_stop_rotation"
                elif iter_step > EARLY_STOP_STEPS:
                    action = {"action": 0}
                    early_stop_reason = "early_stop_steps"
                else:
                    early_stop_reason = agent.early_stop_reason
                
                iter_step+=1
                obs = env.step(action)
                
             
            info = env.get_metrics()
            result_dict = dict()
            result_dict = {k: info[k] for k in target_key if k in info}
            result_dict["id"] = env.current_episode.episode_id
            result_dict["early_stop_reason"] = early_stop_reason
            count+=1

            with open(os.path.join(os.path.join(result_path, "log"),"stats_{}_{}.json".format(env.current_episode.episode_id, i)), "w") as f:
                json.dump(result_dict, f, indent=4)
                
            extra_info = {
                "conversations": [item for item in agent.conversations if item['role'] == 'assistant'],
                # "locations": env._task.measurements.measures['ndtw'].locations
            }
            with open(os.path.join(os.path.join(result_path, "extra_info"),"info_{}_{}.json".format(env.current_episode.episode_id, i)), "w") as f:
                json.dump(extra_info, f)
            
            
            t_dict["t_episode"] = time.time() - episode_start_time
            result_queue.put(t_dict)
        
        env.close()

class ActiveVlnAgent:
    def __init__(self, action_space, api_key, base_url, result_path, 
                    max_pixels, max_turns, num_generations = 1, require_map=False):
            
        self.result_path = result_path
        self.require_map = require_map
        self.max_pixels = max_pixels
        self.num_generations = num_generations
        self.action_space = action_space
        assert self.action_space in ["r2r", "rxr"], "action_space must be either 'r2r' or 'rxr'"
        self.forward_distance = 25
        self.turn_angle = 15 if self.action_space == "r2r" else 30
        self.max_turns = max_turns
        
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "extra_info"), exist_ok=True)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id
        
        self.sampling_params = {
            "n": 1,
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 0.8,
        }
        
        self.history_rgb_tensor = None
        
        self.topdown_map_list = []
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R if self.action_space == "r2r" else SYSTEM_PROMPT_RxR}]})

        self.count_id = 0
        self.count_turn = 0
        self.early_stop_reason = None
        self.reset()

    def predict_inference(self):     
        outputs = self.client.chat.completions.create(
            messages=self.conversations,
            model=self.model,
            max_completion_tokens=self.sampling_params["max_tokens"],
            temperature=self.sampling_params["temperature"],
            top_p=self.sampling_params["top_p"],
        )
        output_text = outputs.choices[0].message.content
        output_text = output_text.strip()
        
        return output_text

    def extract_multi_result(self, output):
        sub_actions = output.split(', ')
        result = []
        for sub_action in sub_actions:
            action_index, numeric = self.extract_result(sub_action)
            result.append([action_index, numeric])
        return result

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output = output.strip().lower()
        
        if "stop" in output:
            return 0, None
        elif "forward" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 1, self.forward_distance
            match = match.group()
            return 1, float(match)
        elif "left" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 2, self.turn_angle
            match = match.group()
            return 2, float(match)
        elif "right" in output:
            match = re.search(r'-?\d+', output)
            if match is None:
                return 3, self.turn_angle
            match = match.group()
            return 3, float(match)
        return None, None
    

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instuction.split(' ')
        max_width = new_image.shape[1]
        x = 10
        line = ""

        for word in words:

            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line ), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1]+5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image

    def action_id_to_str(self,action_id):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
        if action_id == 0:
            return "stop"
        elif action_id == 1:
            return "forward"
        elif action_id == 2:
            return "turn left"
        elif action_id == 3:
            return "turn right"
        else:
            raise ValueError(f"Invalid action ID: {action_id}")
        
    def reset(self):       
        if self.require_map:
            if len(self.topdown_map_list)!=0:
                output_video_path = os.path.join(self.result_path, "video","{}.gif".format(self.episode_id))

                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_tensor = None
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.count_turn = 0
        self.early_stop_reason = None

        self.pending_action_list = []

        self.first_forward = False
        self.conversations = []
        self.conversations.append({
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_R2R if self.action_space == "r2r" else SYSTEM_PROMPT_RxR}]})
        
    def act(self, observations, info, episode_id):

        self.episode_id = episode_id
        rgb = observations["rgb"]

        rgb_ = Image.fromarray(rgb.astype('uint8')).convert('RGB')
        resized_height, resized_width = smart_resize(
            rgb_.height,
            rgb_.width,
            max_pixels=self.max_pixels,
            factor=28,
        )
        rgb_ = rgb_.resize((resized_width, resized_height))
  
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            output_im = np.concatenate((rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0 :
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], "Pending action: {}".format(temp_action))
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}

        self.count_turn += 1
        if self.count_turn > self.max_turns:
            print("Forced to stop as maximum number of turns reached.")
            self.early_stop_reason = "max_turns_reached"
            return {"action": 0}
        
        content = []

        s = "[Initial Observation]:" if len(self.conversations) == 1 else "After that, the observation is:"
        content.append({"type": "text", "text": s})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image_to_base64(rgb_)}"}})
        item = first_turn_user_prompt if len(self.conversations) == 1 else normal_user_prompt
        content.append({"type": "text", "text": item.format(observations["instruction"]["text"])})

        self.conversations.append({
                "role": "user",
                "content": content
            })

        try:
            navigation = self.predict_inference()
        except Exception as e:
            print(f"Error during inference: {e}")
            self.early_stop_reason = "inference_error"
            return {"action": 0}
            
        self.conversations.append({
            "role": "assistant",
            "content": [{"type": "text", "text": navigation}]
        })
        
        if self.require_map:
            img = self.addtext(output_im, observations["instruction"]["text"], navigation)
            self.topdown_map_list.append(img)

        result = self.extract_multi_result(navigation)
        for action_index,numeric in result:
            if action_index == 0:
                self.pending_action_list.append(0)
            elif action_index == 1:
                for _ in range(min(3, int(numeric/self.forward_distance))):
                    self.pending_action_list.append(1)

            elif action_index == 2:
                for _ in range(min(3,int(numeric/self.turn_angle))):
                    self.pending_action_list.append(2)

            elif action_index == 3:
                for _ in range(min(3,int(numeric/self.turn_angle))):
                    self.pending_action_list.append(3)
            
            if action_index is None or len(self.pending_action_list)==0:
                print('select a random action')
                action_index = random.randint(1, 3)
                navigation = self.action_id_to_str(action_index)
                self.pending_action_list.append(action_index)

        return {"action": self.pending_action_list.pop(0)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-config",type=str, required=True, help="Path to the VLN-CE baseline configuration file.")
    parser.add_argument("--split-num",type=int, required=True, help="Number of evaluation chunks to divide the task into.")
    parser.add_argument("--result-path",type=str, required=True, help="Directory to save the evaluation results.")
    parser.add_argument("--action-space",type=str, choices=["r2r", "rxr"], default="r2r", help="Choose the action space to use: r2r or rxr.")
    parser.add_argument("--num-generations",type=int, help="Number of times to repeat the sampling process.", default=1)
    parser.add_argument("--max-turns", type=int, default=120, help="Maximum number of turns allowed in a single episode.")
    parser.add_argument("--max-pixels",type=int, default=76800)

    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE")
    assert api_key is not None and base_url is not None

    config = get_config(args.exp_config)
    dataset = habitat.datasets.make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)

    dataset_splits = dataset.get_splits(args.split_num,allow_uneven_splits=True)
    num_episodes = len(dataset.episodes) 

    manager = mp.Manager()
    result_queue = manager.Queue()
    processes = []
    for i in range(args.split_num):
        worker_args = (result_queue, args.action_space, api_key, base_url, config, dataset_splits[i], args.result_path,
                args.num_generations,
                args.max_pixels, args.max_turns)
        p = mp.Process(target=evaluate_agent, args=worker_args, daemon=True)
        p.start()
        processes.append(p)

    with tqdm(total=num_episodes*args.num_generations, desc="Evaluating") as pbar:
        for _ in range(num_episodes):
            result = result_queue.get()
            pbar.update(1)
            pbar.set_postfix(**result)
    
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    main()