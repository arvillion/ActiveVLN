import re
from typing import Dict
from ..constants import STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT

def parse_action(action, action_space):
    """
    Parse the action string into a tuple of (action_name, action_value).
    """
    action_pattern = {
        "r2r": rf'^(({MOVE_FORWARD}) (25|50|75)cm|(({TURN_LEFT})|({TURN_RIGHT})) (15|30|45) degrees|({STOP}))$',
        "rxr": rf'^(({MOVE_FORWARD}) (25|50|75)cm|(({TURN_LEFT})|({TURN_RIGHT})) (30|60|90) degrees|({STOP}))$',
    }
    action = action.strip()
    if action_space not in action_pattern:
        raise ValueError(f"Unknown data source: {action_space}")
    action_regex = action_pattern[action_space]
    action_match = re.match(action_regex, action)
    if action_match is None:
        raise ValueError(f"Unknown action: {action}")
    
    if action.startswith(MOVE_FORWARD):
        return (MOVE_FORWARD, int(action.split(" ")[-1].replace("cm", "")))
    elif action.startswith(TURN_LEFT):
        return (TURN_LEFT, int(action.split(" ")[-2]))
    elif action.startswith(TURN_RIGHT):
        return (TURN_RIGHT, int(action.split(" ")[-2]))
    else:
        return (STOP, None)


def parse_no_think_no_tag(response: str, special_token_list=None, action_sep=',', max_actions=3) -> Dict:
    """
    Parse response in format: <answer>...</answer>
    
    Returns a dict with keys:
    - llm_raw_response: the original response
    - llm_response: the response with <answer> tag
    - think_content: empty string (no think content in this format)
    - action_content: the content inside <answer> tag
    - actions: a list of actions extracted from action_content
    - format_correct: whether the response strictly follows the expected format
    """
    response = response.replace("<image>","")
        
    action_content = response.strip()
    think_content = ""  # No think content in this format
    if special_token_list is not None:
        for special_token in special_token_list:
            action_content = action_content.replace(special_token, "").strip()
    actions = [action.strip() for action in action_content.split(action_sep) if action.strip()]
    
    format_correct = len(actions) <= max_actions
    
    if len(actions) > max_actions:
        actions = actions[:max_actions]
        action_content = (" " + action_sep + " ").join(actions)

    llm_response = action_content.strip()
    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct
    }

    
PARSE_FUNC_MAP = {
    "no_think_no_tag": parse_no_think_no_tag,
}
