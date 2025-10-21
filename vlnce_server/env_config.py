from dataclasses import dataclass, fields,field
from typing import Optional, List
import hashlib
import json
from .constants import REWARD_SUCCESS

@dataclass
class VLNCEEnvConfig:
    # properties below should have the default value
    env_name: str = "vlnce"
    prompt_format: str = "no_think" 
    max_actions_per_step: int = 3 
    action_sep: str = ","
    image_placeholder: str = "<image>"
    special_token_list: Optional[List[str]] = field(default_factory=lambda: ["<think>", "</think>", "<answer>", "</answer>"])

    # episode dependent properties
    episode_id: int = 1
    history_actions: List[str] = field(default_factory=list)
    data_source: str = "r2r"
    action_space: str = "r2r"
    step_budget: int = 0
    turn_budget: int = 0
    
    # reward settings
    reward_type: str = REWARD_SUCCESS
    success_reward_base: float = 10.0
    ndtw_reward_base: float = 5.0
    format_reward: float = 0

    # utility settings
    experiment_name: str = "expr"
    save_as_video: bool = False

    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def config_id(self) -> str:
        id_fields=["episode_id", "history_actions", "data_source"]
        id_str = {field.name: getattr(self, field.name) for field in fields(self) if field.name in id_fields}
        id_str = hashlib.sha256(json.dumps(id_str, sort_keys=True).encode()).hexdigest()
        return f"VLNCEEnvConfig({id_str})"

    @property
    def cache_key(self):
        cache_fields = ["data_source", "episode_id"]
        id_str = {field.name: getattr(self, field.name) for field in fields(self) if field.name in cache_fields}
        id_str = hashlib.sha256(json.dumps(id_str, sort_keys=True).encode()).hexdigest()
        return str(id_str)

if __name__ == "__main__":
    config = VLNCEEnvConfig()
    print(config.config_id())