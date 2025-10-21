SYSTEM_PROMPT_NO_THINK = {
    "r2r": (
        "You are a helpful assistant. "
        "Your goal is to follow the given instruction to reach a specified destination. \n"
        "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
        "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
        "The instruction will be provided with each observation. You can take multiple actions at each turn. "
    ),
    "rxr": (
        "You are a helpful assistant. "
        "Your goal is to follow the given instruction to reach a specified destination. \n"
        "At each step, you receive a first-person image (starting view if first step (step 1), or post-action view otherwise). "
        "Your task is to select choose one action from: move forward 25cm, move forward 50cm, move forward 75cm, turn left 30 degrees, turn left 60 degrees, turn left 90 degrees, turn right 30 degrees, turn right 60 degrees, turn right 90 degrees, or stop. \n"
        "The instruction will be provided with each observation. You can take multiple actions at each turn. "
    ),
}

def init_observation_template(observation, instruction):
    return f"""[Initial Observation]:
{observation}
Instruction: {instruction}
Decide your next action. """

def action_template(observation, instruction):
    return f"""After that, the observation is:
{observation}
Instruction: {instruction}
Decide your next action. """


FORMAT_CONFIGS = {

    "no_think_no_tag": {
        "format": None,
        "description": None,
        "example": None,
        "system_prompt": SYSTEM_PROMPT_NO_THINK,
    }
}


def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified robot navigation format type.
    This returned function creates the per-turn instruction for the LLM.
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format for the robot navigation task.
        
        Args:
            max_actions_per_step (int): Max actions. Defaults to 5 (common for robot).
            action_sep (str): Separator. Defaults to ',' (common for robot).
            add_example (bool): Whether to add an example. Defaults to True.
            
        Returns:
            str: The formatted prompt.
        """
        # Defaults suitable for the robot navigation task
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True) # Default to True as per robot examples
        
        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
{config["description"]}"""
        
        if "additional_info" in config: # In case it's added to FORMAT_CONFIGS later
            base_prompt += f"\n{config['additional_info']}"
        
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        if add_example:
            # The 'e.g.' is already part of the example string in this FORMAT_CONFIGS
            example_text = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example_text}"
        
        return base_prompt
    
    return prompt_function

def format_prompt_generator(format_type):
    # format_prompt = base_prompt + description + response_format + example
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format.
        
        Args:
            add_example (bool): Whether to add an example
            
        Returns:
            str: The formatted prompt
        """
        max_actions_per_step = kwargs.get("max_actions_per_step", 3)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", False)

        config = FORMAT_CONFIGS[format_type]
        
        if max_actions_per_step > 1:
            base_prompt = f"You can take up to {max_actions_per_step} actions at a time, separated by '{action_sep}'. "
        else:
            base_prompt = "You can only take one action at a time. "
        
        if config["description"]:
            base_prompt += f"{config['description']}"
        
        # Add response format instruction
        if config["format"]:
            base_prompt += f"""
    Your response should be in the format of:
    {config["format"]} """
        
        # Add example if requested
        if add_example:
            example = config["example"]
            return base_prompt + '\n' + f"e.g. {example}"
        
        return base_prompt
    
    return prompt_function

# Generate the format prompt dictionary using the generator
format_prompt = {format_type: format_prompt_generator(format_type) 
                for format_type in FORMAT_CONFIGS}

if __name__ == "__main__":
    # Example usage
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(add_example=False, max_actions_per_step=1, action_sep=","))
        print("\n" + "="*50 + "\n")