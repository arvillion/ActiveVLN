import ray
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import uuid
import asyncio

from .env import Simulator, VLNCEEnv
from .env_config import VLNCEEnvConfig
from .service_config import VLNCEServiceConfig
from .utils.serial_utils import serialize_observation

@ray.remote
class ActorPool:
    def __init__(self, logical_to_physical, gpu_plan, resource_prefix):
        self.logical_to_physical = logical_to_physical
        self.gpu_plan = gpu_plan
        self.condition = asyncio.Condition()
        self.resources = {} # id->simulator. Each resource(simulator) has a unique id
        self.in_use = set() # records the id of in-use resources
        self.free_by_key = defaultdict(deque)  # cache_key -> deque of resource ids
        self.resource_prefix = resource_prefix
        
        self.max_in_use = 0
        asyncio.get_event_loop().create_task(self._periodic_print_status())
             

    async def init_pool(self, dataset):
        async with self.condition:
            initial_cache_key = uuid.uuid4() # 统一初始化为一个随机的cache_key
            for logical_gpu_id, count in enumerate(self.gpu_plan):
                physical_gpu_id = self.logical_to_physical[logical_gpu_id]
                for _ in range(count):
                    worker = Simulator.options(lifetime="detached", resources={f"{self.resource_prefix}_gpu{logical_gpu_id}": 1}).remote(physical_gpu_id, dataset)
                    rid = self.resource_prefix + str(uuid.uuid4())
                    self.resources[rid] = worker
                    self.free_by_key[initial_cache_key].append(rid)
                    
            self.condition.notify_all()
                
    async def acquire_many(self, cache_keys: List[str]):
        if len(cache_keys) == 0:
            return []
        async with self.condition:
            while True:
                acquired = [] # list of (rid, original_cache_key)
                # used_ids = set()
                fallback_keys = []

                for key in cache_keys:
                    queue = self.free_by_key[key]
                    if queue:
                        rid = queue.popleft()
                        self.in_use.add(rid)
                        acquired.append((rid, key))
                    else:
                        acquired.append((None, None))
                        fallback_keys.append(key)

                # Try to fulfill fallback keys with any remaining free resources
                if fallback_keys:
                    # Available fallback pool: all non-empty queues from other keys
                    for i, (rid, key) in enumerate(acquired):
                        if rid is not None:
                            continue
                        for alt_key, queue in self.free_by_key.items():
                            if queue:
                                rid = queue.popleft()
                                self.in_use.add(rid)
                                acquired[i] = (rid, alt_key)
                                break

                # Check if fully satisfied
                if all(rid is not None for rid, _ in acquired):
                    return [(rid, self.resources[rid]) for rid, _ in acquired]

                # Rollback all partial acquisitions
                for rid, key in acquired:
                    if rid is not None:
                        self.in_use.remove(rid)
                        self.free_by_key[key].appendleft(rid)

                await self.condition.wait()
            
    async def release_many(self, resource_ids, cache_keys):
        assert len(resource_ids) == len(cache_keys)
        async with self.condition:
            for rid, key in zip(resource_ids, cache_keys):
                assert rid in self.in_use
                if rid in self.in_use:
                    self.in_use.remove(rid)
                    self.free_by_key[key].append(rid)
            self.condition.notify_all()
        
    def get_resource_prefix(self):
        return self.resource_prefix
    
    async def _periodic_print_status(self, interval: int = 10):
        while True:
            await asyncio.sleep(interval)
            self.print_resource_status()
    
    def print_resource_status(self):
        total = len(self.resources)
        in_use = len(self.in_use)
        free = total - in_use
        self.max_in_use = max(self.max_in_use, in_use)
        print(f"[ActorPool:{self.resource_prefix}] Total: {total}, In use: {in_use}, Free: {free}, Max in use: {self.max_in_use}")
    
@ray.remote
class VLNCEActor:
    def __init__(self, cfg: VLNCEEnvConfig, sim, save_video_dir: str):
        self.env_config = cfg
        self.env = VLNCEEnv(cfg, sim, save_video_dir)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(seed=seed)
        return serialize_observation(obs), info

    def step(self, action: Any) -> Tuple[Dict, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        return serialize_observation(obs), reward, done, info

    def get_system_prompt(self) -> str:
        return self.env.system_prompt()
    
    def compute_reward(self):
        return self.env.compute_reward()

    def close(self):
        self.env.close()

@ray.remote
class VLNCEService:
    def __init__(self, config: VLNCEServiceConfig, r2r_actor_pool, rxr_actor_pool):
        self.config = config
        self.actors = {}
        self.sims = {}
        
        self.r2r_actor_pool = r2r_actor_pool
        self.rxr_actor_pool = rxr_actor_pool
        
        self.r2r_resource_prefix = ray.get(self.r2r_actor_pool.get_resource_prefix.remote())
        self.rxr_resource_prefix = ray.get(self.rxr_actor_pool.get_resource_prefix.remote())



    def create_environments_batch(self, ids2configs: Dict[str, VLNCEEnvConfig]) -> None:
    
        id_to_cfgs = {id: VLNCEEnvConfig(**cfg['env_config']) for id, cfg in ids2configs.items()}
        id_to_cfgs_by_data_source = defaultdict(dict)
        for id, cfg in id_to_cfgs.items():
            assert cfg.data_source in ['r2r', 'rxr'], f"Unsupported data source: {cfg.data_source}"
            id_to_cfgs_by_data_source[cfg.data_source].update({ id: cfg })
        
        r2r_cache_keys = [cfg.cache_key for cfg in id_to_cfgs_by_data_source['r2r'].values()]
        rxr_cache_keys = [cfg.cache_key for cfg in id_to_cfgs_by_data_source['rxr'].values()]
        r2r_sims, rxr_sims = ray.get([
            self.r2r_actor_pool.acquire_many.remote(r2r_cache_keys),
            self.rxr_actor_pool.acquire_many.remote(rxr_cache_keys)
        ])
    
        futures = {}
        for (env_id, cfg), cache_key, (rid, sim) in zip(id_to_cfgs_by_data_source['r2r'].items(), r2r_cache_keys, r2r_sims):
            actor = VLNCEActor.remote(cfg, sim, self.config.save_video_dir)
            futures[env_id] = actor
            self.sims[env_id] = (rid, cache_key, sim)
        
        for (env_id, cfg), cache_key, (rid, sim) in zip(id_to_cfgs_by_data_source['rxr'].items(), rxr_cache_keys, rxr_sims):
            actor = VLNCEActor.remote(cfg, sim, self.config.save_video_dir)
            futures[env_id] = actor
            self.sims[env_id] = (rid, cache_key, sim)
        
        # wait for all actors to be created
        self.actors.update(futures)
        
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        
        futures = {
            env_id: self.actors[env_id].reset.remote(seed)
            for env_id, seed in ids2seeds.items()
        }
        results = {
            env_id: future for env_id, future in futures.items()
        }
        
        return results

    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        
        results = {
            env_id: self.actors[env_id].step.remote(action)
            for env_id, action in ids2actions.items()
        }
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        
        
        if env_ids is None:
            env_ids = list(self.actors.keys())
        
        resource_ids = [self.sims[env_id][0] for env_id in env_ids]
        cache_keys = [self.sims[env_id][1] for env_id in env_ids]
        
        r2r_resource_ids = []
        r2r_cache_keys = []
        rxr_resource_ids = []
        rxr_cache_keys = []
        
        for rid, cache_key in zip(resource_ids, cache_keys):
            if rid.startswith(self.r2r_resource_prefix):
                r2r_resource_ids.append(rid)
                r2r_cache_keys.append(cache_key)
            elif rid.startswith(self.rxr_resource_prefix):
                rxr_resource_ids.append(rid)
                rxr_cache_keys.append(cache_key)
            else:
                raise ValueError(f"Resource ID {rid} does not match any known prefix.")
        
        ray.get([
            self.r2r_actor_pool.release_many.remote(r2r_resource_ids, r2r_cache_keys),
            self.rxr_actor_pool.release_many.remote(rxr_resource_ids, rxr_cache_keys)
        ])
        
        for env_id in env_ids:
            self.actors.pop(env_id, None)
            self.sims.pop(env_id, None)
            
    
    def warmup_batch(self, cfgs) -> None:
        
        assert all(cfg['data_source'] in ['r2r', 'rxr'] for cfg in cfgs)
        
        cfgs = [VLNCEEnvConfig(**cfg) for cfg in cfgs]
        cfgs_by_data_source = defaultdict(list)
        for cfg in cfgs:
            cfgs_by_data_source[cfg.data_source].append(cfg)

        r2r_cfgs = cfgs_by_data_source['r2r']
        r2r_cache_keys = [cfg.cache_key for cfg in r2r_cfgs]
        r2r_sims = ray.get(self.r2r_actor_pool.acquire_many.remote(r2r_cache_keys)) if len(r2r_cache_keys) > 0 else []
        r2r_resource_ids = [sim[0] for sim in r2r_sims]
        
        rxr_cfgs = cfgs_by_data_source['rxr']
        rxr_cache_keys = [cfg.cache_key for cfg in rxr_cfgs]
        rxr_sims = ray.get(self.rxr_actor_pool.acquire_many.remote(rxr_cache_keys)) if len(rxr_cache_keys) > 0 else []
        rxr_resource_ids = [sim[0] for sim in rxr_sims]
        
        for cfg, (rid, sim) in zip(r2r_cfgs, r2r_sims):
            sim.start_new_episode.remote(cfg.episode_id)
        
        for cfg, (rid, sim) in zip(rxr_cfgs, rxr_sims):
            sim.start_new_episode.remote(cfg.episode_id)

        ray.get([
            self.r2r_actor_pool.release_many.remote(r2r_resource_ids, r2r_cache_keys),
            self.rxr_actor_pool.release_many.remote(rxr_resource_ids, rxr_cache_keys),
        ])
