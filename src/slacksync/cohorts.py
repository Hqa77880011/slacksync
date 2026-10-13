import copy

from .kitchen import Kitchen
from .utils import load, save


def make_cohort(source, output, cohort):
    episodes = []
    for index, original in enumerate(load(source)):
        episode = copy.deepcopy(original)
        episode["cohort"] = cohort
        episode["controlled"] = cohort in ("controlled", "extreme_shift", "congestion")
        episode["action_duration"] = (
            (1, 2, 4)[index % 3] if cohort == "controlled" else 1
        )
        episode["blocked_cell"] = None
        if episode["controlled"]:
            kitchen = Kitchen(episode["layout"], episode["initial_state"])
            floor = sorted(kitchen.mdp.get_valid_player_positions())
            episode["blocked_cell"] = list(floor[len(floor) // 2])
        if cohort == "extreme_shift":
            episode["controller"] = "burst_stay"
            episode["pause_exit"] = 0.1
            episode["action_duration"] = (1, 4, 8)[index % 3]
        if cohort == "calibration_controller":
            episode["controller"] = "clockwise"
        episodes.append(episode)
    save(output, episodes)
