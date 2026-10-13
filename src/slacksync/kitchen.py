from collections import deque

import numpy as np


SKILLS = (
    "fetch_onion",
    "fetch_tomato",
    "put_onion",
    "put_tomato",
    "start_cooking",
    "wait_pot",
    "fetch_dish",
    "collect_soup",
    "serve",
    "stage_onion",
    "pickup_staged",
    "drop_object",
    "pickup_object",
)
DIRECTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))


def held(player):
    return player.held_object.name if player.held_object is not None else "empty"


class Kitchen:
    def __init__(
        self,
        layout,
        initial=None,
        controlled=False,
        seed=17,
        blocked=None,
        duration=1,
        controller="native",
        pause_exit=0.5,
    ):
        from overcooked_ai_py.mdp.overcooked_mdp import (
            OvercookedGridworld,
            OvercookedState,
        )

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        self.state = (
            OvercookedState.from_dict(initial)
            if initial
            else self.mdp.get_standard_start_state()
        )
        self.controlled = controlled
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.pause_exit = pause_exit
        self.paused = False
        self.reroute = False
        self.blocked = tuple(blocked) if blocked else None
        self.duration = duration
        if controller not in (
            "native",
            "cautious",
            "stochastic",
            "lagged",
            "lex",
            "random_tie",
            "clockwise",
            "counterclockwise",
            "reactive_yield",
            "persistent_reroute",
            "burst_stay",
        ):
            raise ValueError("Unknown teammate controller family")
        self.controller = controller
        self.tick = 0
        self.soups = 0
        self.collisions = [0, 0]
        self.cooldown = [0, 0]
        self.last_info = {}
        self.stage_completed = set()

    def stations(self):
        return {
            kind: [list(p) for p in positions]
            for kind, positions in self.mdp.terrain_pos_dict.items()
        }

    def neighbors(self, position):
        directions = list(DIRECTIONS)
        if self.controller == "clockwise":
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        elif self.controller == "counterclockwise":
            directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        elif self.controller == "random_tie":
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.tick, *map(int, position)])
            )
            rng.shuffle(directions)
        elif self.controller == "persistent_reroute" and self.reroute:
            directions.reverse()
        for direction in directions:
            point = tuple(np.asarray(position) + direction)
            if point in self.mdp.get_valid_player_positions():
                yield point, direction

    def path(self, start, targets, occupied=()):
        targets = set(map(tuple, targets))
        queue = deque([(tuple(start), [])])
        seen = {tuple(start)}
        obstacles = set(map(tuple, occupied))
        if self.controlled and self.blocked and 160 <= self.tick < 180:
            obstacles.add(self.blocked)
        while queue:
            point, route = queue.popleft()
            if point in targets:
                return route
            for next_point, direction in self.neighbors(point):
                if next_point not in seen and next_point not in obstacles:
                    seen.add(next_point)
                    queue.append((next_point, route + [direction]))
        return None

    def approaches(self, station):
        result = []
        for direction in DIRECTIONS:
            position = tuple(np.asarray(station) - direction)
            if position in self.mdp.get_valid_player_positions():
                result.append((position, direction))
        return result

    def travel(self, owner, station):
        targets = [p for p, _ in self.approaches(station)]
        route = self.path(self.state.players[owner].position, targets)
        return len(route) + 1 if route is not None else float("inf")

    def move_to(self, owner, station, occupied=True):
        player = self.state.players[owner]
        approaches = self.approaches(station)
        positions = [p for p, _ in approaches]
        other = (
            [p.position for i, p in enumerate(self.state.players) if i != owner]
            if occupied
            else []
        )
        route = self.path(player.position, positions, other)
        if route is None:
            return (0, 0)
        if route:
            return route[0]
        orientation = next(
            direction for point, direction in approaches if point == player.position
        )
        return orientation if player.orientation != orientation else "interact"

    def ready(self, job):
        player, station = self.state.players[job["agent"]], tuple(job["station"])
        kind = job["skill"]
        object_at = self.state.objects.get(station)
        if kind in ("fetch_onion", "fetch_tomato", "fetch_dish"):
            return held(player) == "empty"
        if kind == "stage_onion":
            return held(player) == "empty" and object_at is None
        if kind == "pickup_staged":
            return (
                held(player) == "empty"
                and object_at is not None
                and object_at.name == "onion"
            )
        if kind in ("put_onion", "put_tomato"):
            ingredient = "onion" if kind == "put_onion" else "tomato"
            return held(player) == ingredient and (
                object_at is None or not object_at.is_full
            )
        if kind == "drop_object":
            return held(player) == job["object"] and object_at is None
        if kind == "pickup_object":
            return (
                held(player) == "empty"
                and object_at is not None
                and object_at.name == job["object"]
            )
        if kind == "start_cooking":
            return (
                held(player) == "empty"
                and object_at is not None
                and object_at.name == "soup"
                and object_at.is_idle
                and len(object_at.ingredients) > 0
            )
        if kind == "wait_pot":
            return (
                object_at is not None
                and object_at.name == "soup"
                and object_at.is_ready
            )
        if kind == "collect_soup":
            return (
                held(player) == "dish"
                and object_at is not None
                and object_at.name == "soup"
                and object_at.is_ready
            )
        if kind == "serve":
            return held(player) == "soup"
        return False

    def begin(self, job):
        return {
            "held": held(self.state.players[job["agent"]]),
            "soups": self.soups,
            "object": repr(self.state.objects.get(tuple(job["station"]))),
            "started": self.tick,
        }

    def complete(self, job, before):
        player = self.state.players[job["agent"]]
        kind, station = job["skill"], tuple(job["station"])
        obj = self.state.objects.get(station)
        if kind in ("fetch_onion", "pickup_staged"):
            return held(player) == "onion"
        if kind == "fetch_tomato":
            return held(player) == "tomato"
        if kind == "pickup_object":
            return held(player) == job["object"]
        if kind == "drop_object":
            return (
                before["held"] == job["object"]
                and held(player) == "empty"
                and obj is not None
                and obj.name == job["object"]
            )
        if kind == "fetch_dish":
            return held(player) == "dish"
        if kind in ("put_onion", "put_tomato"):
            ingredient = "onion" if kind == "put_onion" else "tomato"
            return (
                before["held"] == ingredient
                and held(player) == "empty"
                and obj is not None
            )
        if kind == "start_cooking":
            return obj is not None and (obj.is_cooking or obj.is_ready)
        if kind == "wait_pot":
            return self.ready(job)
        if kind == "collect_soup":
            return held(player) == "soup"
        if kind == "serve":
            return self.soups > before["soups"] and held(player) == "empty"
        if kind == "stage_onion":
            return (
                held(player) == "empty"
                and obj is not None
                and obj.name == "onion"
                and self.tick > before["started"]
            )
        return False

    def action(self, job):
        owner, kind = job["agent"], job["skill"]
        if kind == "wait_pot":
            return (0, 0)
        if kind == "stage_onion" and held(self.state.players[owner]) == "empty":
            return self.move_to(owner, job["supply"])
        return self.move_to(owner, job["station"])

    def resume_action(self, owner, resume):
        player = self.state.players[owner]
        if self.resume_ready(owner, resume):
            return (0, 0)
        route = self.resume_route(
            player.position,
            resume,
            [p.position for i, p in enumerate(self.state.players) if i != owner],
        )
        if route is None:
            return (0, 0)
        if route:
            return route[0]
        return (0, 0)

    def resume_route(self, start, resume, occupied=()):
        target = tuple(resume["position"])
        orientation = tuple(resume["orientation"])
        forward = tuple(np.asarray(target) + orientation)
        if forward not in self.mdp.get_valid_player_positions():
            route = self.path(start, [target], occupied)
            return route + [orientation] if route is not None else None
        behind = tuple(np.asarray(target) - orientation)
        if behind not in self.mdp.get_valid_player_positions():
            return None
        route = self.path(start, [behind], occupied)
        return route + [orientation] if route is not None else None

    def valid_interaction(self, owner, job):
        player = self.state.players[owner]
        target = tuple(np.asarray(player.position) + player.orientation)
        expected = (
            tuple(job["supply"])
            if job["skill"] == "stage_onion" and held(player) == "empty"
            else tuple(job["station"])
        )
        if target != expected:
            return False
        if job["skill"] == "stage_onion" and held(player) == "onion":
            return target not in self.state.objects
        return self.ready(job)

    def resume_ready(self, owner, resume):
        player = self.state.players[owner]
        return (
            player.position == tuple(resume["position"])
            and player.orientation == tuple(resume["orientation"])
            and held(player) == resume["held"]
        )

    def step(self, actions):
        old = [p.position for p in self.state.players]
        executed = list(actions)
        event_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.tick, 719])
        )
        if self.controller == "persistent_reroute" and self.collisions[1] >= 2:
            self.reroute = not self.reroute
        if self.controller == "burst_stay":
            if self.paused:
                self.paused = event_rng.random() >= self.pause_exit
            else:
                self.paused = event_rng.random() < 0.1
            if self.paused:
                executed[1] = (0, 0)
        if self.controller == "stochastic" and event_rng.random() < 0.15:
            executed[1] = (0, 0)
        elif self.controller == "lagged" and self.tick % 3 == 0:
            executed[1] = (0, 0)
        elif (
            self.controller in ("cautious", "reactive_yield") and self.collisions[1] > 0
        ):
            executed[1] = (0, 0)
        for owner, action in enumerate(executed):
            if self.controlled and self.cooldown[owner] > 0:
                executed[owner] = (0, 0)
                self.cooldown[owner] -= 1
            elif self.controlled and action != (0, 0):
                self.cooldown[owner] = self.duration - 1
            if (
                self.controlled
                and self.blocked
                and 160 <= self.tick < 180
                and isinstance(executed[owner], tuple)
            ):
                if tuple(np.asarray(old[owner]) + executed[owner]) == self.blocked:
                    executed[owner] = (0, 0)
        state, info = self.mdp.get_state_transition(self.state, tuple(executed))
        self.state, self.last_info = state, info
        self.tick += 1
        self.soups += sum(
            bool(delivered) and reward > 0
            for delivered, reward in zip(
                info["event_infos"].get("soup_delivery", [False, False]),
                info["sparse_reward_by_agent"],
            )
        )
        for owner, action in enumerate(executed):
            attempted_floor = (
                isinstance(action, tuple)
                and tuple(np.asarray(old[owner]) + action)
                in self.mdp.get_valid_player_positions()
            )
            self.collisions[owner] = (
                min(10, self.collisions[owner] + 1)
                if attempted_floor
                and action != (0, 0)
                and self.state.players[owner].position == old[owner]
                else max(0, self.collisions[owner] - 1)
            )
        return executed

    def features(self, owner, head):
        width, height = self.mdp.width, self.mdp.height
        values = []
        for player in self.state.players:
            values.extend([player.position[0] / width, player.position[1] / height])
            values.extend([float(player.orientation == d) for d in DIRECTIONS])
            values.extend(
                [
                    float(held(player) == name)
                    for name in ("empty", "onion", "tomato", "dish", "soup")
                ]
            )
        station = tuple(head["station"])
        obj = self.state.objects.get(station)
        values.extend(
            [
                len(obj.ingredients) / 3
                if obj is not None and obj.name == "soup"
                else 0.0,
                float(obj.is_cooking)
                if obj is not None and obj.name == "soup"
                else 0.0,
                obj._cooking_tick / max(obj.cook_time, 1)
                if obj is not None and obj.name == "soup" and not obj.is_idle
                else 0.0,
                min(self.travel(owner, station), 100) / 100,
                self.collisions[owner] / 10,
                float(owner),
            ]
        )
        values.extend([float(head["skill"] == skill) for skill in SKILLS])
        return np.asarray(values, dtype=np.float32)


def validate_plan(plan, kitchen):
    from .method import graph_ranks

    vertices = plan["vertices"]
    identifiers = [v["id"] for v in vertices]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicated macro task identifiers")
    terrain = {
        "fetch_onion": "O",
        "fetch_tomato": "T",
        "put_onion": "P",
        "put_tomato": "P",
        "start_cooking": "P",
        "wait_pot": "P",
        "fetch_dish": "D",
        "collect_soup": "P",
        "serve": "S",
        "stage_onion": "X",
        "pickup_staged": "X",
        "drop_object": "X",
        "pickup_object": "X",
    }
    for job in vertices:
        if job["skill"] not in SKILLS or job["agent"] not in (0, 1):
            raise ValueError("Unknown skill or agent assignment")
        station = tuple(job["station"])
        if kitchen.mdp.get_terrain_type_at_pos(station) != terrain[job["skill"]]:
            raise ValueError("Skill target has incompatible terrain")
        if (
            job["skill"] == "stage_onion"
            and kitchen.mdp.get_terrain_type_at_pos(tuple(job["supply"])) != "O"
        ):
            raise ValueError("Staging requires an onion dispenser")
        resource = "station:" + str(station)
        job["reads"] = [resource] if job["skill"] == "wait_pot" else []
        job["writes"] = (
            []
            if job["skill"] == "wait_pot"
            else [resource, "hand:" + str(job["agent"])]
        )
        if job["skill"] == "stage_onion":
            job["reads"].append("station:" + str(tuple(job["supply"])))
        job["normal_duration"] = max(float(job.get("normal_duration", 8)), 1)
    for owner in range(2):
        chain = plan["chains"][str(owner)]
        expected = {v["id"] for v in vertices if v["agent"] == owner}
        if len(chain) != len(set(chain)) or set(chain) != expected:
            raise ValueError(
                "Preferred chains must contain each assigned job exactly once"
            )
    graph_ranks(vertices)
    return plan
