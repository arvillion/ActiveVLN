
STOP = "stop"
MOVE_FORWARD = "move forward"
TURN_LEFT = "turn left"
TURN_RIGHT = "turn right"

ACTION_LOOKUP = {
    STOP: 0,
    MOVE_FORWARD: 1,
    TURN_LEFT: 2,
    TURN_RIGHT: 3,
}

REASON_FORMAT_MISMATCH = "unexpected format."
REASON_EPISODE_STEPS_EXCEEDED = "number of steps exceeded."
REASON_EPISODE_TURNS_EXCEEDED = "number of turns exceeded."
REASON_SUCCESS = "successfully reached the goal."
REASON_GOAL_NOT_REACHED = "stopped but goal not reached."

REASON_OUTPUT_LENGTH_EXCEEDED = "output length exceeded."

REWARD_SUCCESS = "success"
REWARD_SUCCESS_NDTW = "success_ndtw"
REWARD_WEIGHTED_SUCCESS = "weighted_success"
REWARD_WEIGHTED_SUCCESS_NDTW = "weighted_success_ndtw"
