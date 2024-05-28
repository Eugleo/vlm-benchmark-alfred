import itertools
import json
import random
import uuid
from collections import Counter
from copy import deepcopy
from typing import Optional

import yaml

from alfred import utils
from alfred.object import plausible_action_objects
from alfred.trajectory import HighLevelAction, Trajectory

HIGH_LEVEL_TASK_PROMPT = """
Your task is to describe what you see in each frame, separately, in a list. For each frame, concisely describe the scene, the objects we're holding, and from that deduce what actions have likely been performed since the last frame.
""".strip()


HIGH_LEVEL_TASK_EXAMPLE = """
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- The possible actions that can be performed are: pickup(object, location), put(object, location), heat(object, microwave), cool(object, fridge), clean(object, sink), slice(object), toggle(object), and goto(location). You should formulate your descriptions of what happened since the last frame based on these actions.
- Multiple actions can happen between frames, though often it will just be a combination of goto and one other action.
- If the first frame starts with us already holding an object, it was already in our hands from the beginning. No `pickup` action happens before the first frame.
- You are bound not to recognize some objects correctly, which might hurt your peformance in downstream tasks. Instead of running with your first guess, try to list a few alternatives for an object if you're unsure what it is.
- Whenever we hold a knife, we might be about to slice an object. If we hold a knife, pay very close attention to minute details in objects' appearance. If there are small lines in the object where there previously weren't any, it is likely that we have sliced it.
- Sometimes, when putting down an object, we put it in or onto another object, instead of a container. In that case you should describe the object we put the object in or onto.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. Frame 1
    - Scene: We are near a countertop, or possibly a large table. Multiple objects are visible on the countertop or table, among them a toaster, a butter knife, a laptop, and a mug.
    - Holding: A circular dark object is at the bottom of the screen, suggesting we are holding it. It might be a bowl, a dark plate, or a pot.
    - Actions since last frame: N/A, this is the first frame.
2. Frame 2
    - Scene: Near a sink. This suggests we have been at the countertop, not a table, previously. The sink is empty, but there is a sponge on the edge of the sink.
    - Holding: The circular dark object is still at the bottom of the screen, so we are still holding it. We still aren't sure what it is.
    - Actions since last frame: goto(sink).
3. Frame 3
    - Scene: Still near the sink. The sponge is still on the edge of the sink. There is a round black object in the sink with a long handle. It seems to be a pan.
    - Holding: The circular dark object is no longer at the bottom of the screen, and we are not holding anything. This means we put it somewhere, probably in the sink.
    - Actions since last frame: We have put the circular dark object in the sink. Since there appears to be a pan in the sink, we now know that the circular dark object has been a pan, not a bowl or a pot. We have done put(pan, sink).
4. Frame 4
    - Scene: Still near the sink. The water is running, suggesting we are cleaning the pan.
    - Holding: We are not holding anything.
    - Actions since last frame: clean(pan, sink).
5. Frame 5
    - Scene: We are now near a bed, or possibly a couch or an armchair. We see a pillow.
    - Holding: It appears that a green rectangular object is lying on the floor near the bottom of the screen, which means we are holding it. It might be a book, or a green sponge. We are not sure where we picked it up.
    - Actions since last frame: goto(bed), pickup(book, *) or possibly pickup(sponge, *). It is likely that we have picked up the green object in the previous location, where we saw a sponge. This would suggest we actually did pickup(sponge, sink) or pickup(sponge, countertop).
"""


class Task:
    prefix: str
    name: str
    prompt_gpt: str
    example_gpt: str
    trajectories: list[Trajectory]
    descriptions: list[str]
    metadata: Optional[dict] = None

    def __init__(
        self,
        prefix,
        name,
        prompt_gpt,
        example_gpt,
        trajectories,
        metadata: Optional[dict] = None,
    ):
        self.prefix = prefix
        self.name = name
        self.trajectories = trajectories
        self.prompt_gpt = prompt_gpt
        self.example_gpt = example_gpt
        self.descriptions = []
        for t in trajectories:
            if t.description not in self.descriptions:
                self.descriptions.append(t.description)
        self.metadata = metadata
        self._check_classes()

    def _check_classes(self):
        if len(self.descriptions) < 2:
            raise ValueError(f"Not enough classes for a task: {self.descriptions}")
        if not all(t.description in self.descriptions for t in self.trajectories):
            raise ValueError("Class missing in descriptions.")

    @property
    def _used_concepts(self):
        objects = set(
            [a.object1 for t in self.trajectories for a in t.actions]
            + [a.object2 for t in self.trajectories for a in t.actions]
        )
        high_level_actions = set(
            [a.action for t in self.trajectories for a in t.actions]
        )
        low_level_actions = set(
            [
                lla.action
                for t in self.trajectories
                for hla in t.actions
                for lla in hla.actions
            ]
        )
        return {
            "objects": [o for o in objects if o],
            "high_level_actions": list(high_level_actions),
            "low_level_actions": list(low_level_actions),
        }

    def write(self, output_dir, max_videos=10, min_frames=32) -> str:
        tasks_dir = output_dir / "tasks" / "alfred" / self.prefix
        tasks_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = output_dir / "videos" / "alfred" / self.prefix / self.name

        description_to_label = {
            d: f"label_{i}" for i, d in enumerate(self.descriptions)
        }
        label_to_description = {v: k for k, v in description_to_label.items()}

        # Task definition
        with open(tasks_dir / f"{self.name}.yaml", "w") as f:
            yaml.dump(
                {
                    "label_prompts": label_to_description,
                    "prompt_gpt": self.prompt_gpt,
                    "example_gpt": self.example_gpt,
                    "metadata": (self.metadata or {})
                    | {"concepts": self._used_concepts},
                },
                f,
            )

        videos = []
        video_counter = Counter()
        # Task videos
        # TODO Could be made more efficient by only having one copy of each video
        random.shuffle(self.trajectories)
        for trajectory in self.trajectories:
            if video_counter[trajectory.description] >= max_videos:
                continue
            video_counter[trajectory.description] += 1
            video_name = f"{trajectory.trial_id}_{uuid.uuid4()}.mp4"
            label = description_to_label[trajectory.description]
            (videos_dir / label).mkdir(parents=True, exist_ok=True)
            video = trajectory.video(min_frames=min_frames)
            video.write_videofile(str(videos_dir / label / video_name), logger=None)
            videos.append(
                {
                    "path": f"{self.prefix}/{self.name}/{label}/{video_name}",
                    "label": label,
                }
            )

        if len(set(video_counter.values())) != 1:
            print(f"WARNING: Unbalanced video classes: {video_counter}")

        # Task data
        with open(tasks_dir / f"{self.name}_data.json", "w") as f:
            json.dump(videos, f)

        return f"{self.prefix}/{self.name}"


def write_config(tasks, output_dir):
    # Config
    with open(output_dir / "configs" / "alfred.yaml", "w") as f:
        config = {
            "tasks": tasks,
            "models": [
                {"kind": "gpt", "model": "gpt-4o", "async_batch": True, "n_frames": 5},
                {"kind": "encoder", "encoder": "s3d", "heads": [{"kind": "cosine"}]},
                {"kind": "encoder", "encoder": "viclip", "heads": [{"kind": "cosine"}]},
                {
                    "kind": "encoder",
                    "encoder": "clip",
                    "heads": [{"kind": "cosine"}],
                    "hf_model": "ViT-bigG-14/laion2b_s39b_b160k",
                    "n_frames": 4,
                },
            ],
            "task_dir": "tasks/alfred",
            "video_dir": "videos/alfred",
            "cache_dir": ".cache",
            "output_dir": "experiments",
        }
        yaml.dump(config, f)


def get_clipped_trajectories(
    trajectories, action, context=5
) -> list[tuple[int, Trajectory]]:
    result = []
    for t in trajectories:
        for idx in [i for i, a in enumerate(t.actions) if a.action == action]:
            hl_actions = deepcopy(t.actions)
            hl_actions[idx - 1].actions = hl_actions[idx - 1].actions[-context:]
            if len(hl_actions) > idx + 1:
                hl_actions[idx + 1].actions = hl_actions[idx + 1].actions[:context]
            beg = (
                idx - 1
                if idx > 0 and hl_actions[idx - 1].action == "GotoLocation"
                else idx
            )
            end = (
                idx + 2
                if idx + 1 < len(hl_actions)
                and hl_actions[idx + 1].action == "GotoLocation"
                else idx + 1
            )
            new_t = t.with_modified_actions(hl_actions[beg:end])
            new_t._description = hl_actions[idx].description.capitalize()
            result.append((idx - beg, new_t))
    return result


def group_trajectories(
    trajectories: list[tuple[int, Trajectory]],
    object1: bool = False,
    object2: bool = False,
) -> list[tuple[str | list[str], str | list[str | None] | None, list[Trajectory]]]:
    return [
        (
            (
                g[0][1].actions[g[0][0]].object1
                if object1
                else [t.actions[idx].object1 for idx, t in g]
            ),
            (
                g[0][1].actions[g[0][0]].object2
                if object2
                else [t.actions[idx].object2 for idx, t in g]
            ),
            [t for _, t in g],
        )
        for g in utils.group_by(
            trajectories,
            lambda t: (
                not object1 or t[1].actions[t[0]].object1,
                not object2 or t[1].actions[t[0]].object2,
            ),
        )
    ]


def prompt_object(container):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us or putting down an object into (or on) a {container}. Your eventual goal will be to recognize the object, but you shouldn't lock-in to one answer too early. Instead, try to describe the object as accurately as possible separately for each frame, refining your answer as you see more frames.
""".strip()


def example_object(container):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- You are bound not to recognize some objects correctly, which might hurt your peformance in downstream tasks. Instead of running with your first guess, try to list a few alternatives for an object if you're unsure what it is.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are in the middle of the room. We do not see the {container} yet.
2. The location has changed. We see something that might be the {container}. We see a multitude of objects there, among them a toaster, a butter knife, a laptop, and a mug, and a pan or possibly a bowl.
2. The scene has not changed much since the last frame. We are still near the {container}.
3. We see a rectangular object at the bottom of the screen, suggesting we are holding it. Considering the selection of objects on the {container} we saw above, it is likely the laptop.
4. The angle has changed a bit and we indeed see that the laptop is still on the {container}. This means the thing we hold must be the toaster.
5. We are now a bit further away from the {container}. We are still holding the toaster.
""".strip()


def object_tasks(trajectories: list[Trajectory]) -> list[Task]:
    all_objects = set(plausible_action_objects["PickupObject"][0])
    object_trajectories = get_clipped_trajectories(trajectories, "PickupObject")

    seen = set()
    tasks = []
    for objects, container, g in group_trajectories(object_trajectories, object2=True):
        assert (container is None or isinstance(container, str)) and isinstance(
            objects, list
        )
        if seen == objects:
            break
        container_trajectories = [
            t for object, t in zip(objects, g) if object not in seen
        ]
        if not container_trajectories:
            continue
        try:
            name = (container or "somewhere").replace(" ", "_")
            tasks.append(
                Task(
                    prefix="foundation/objects",
                    name=f"pick_from_{name}",
                    trajectories=container_trajectories,
                    prompt_gpt=prompt_object(container),
                    example_gpt=example_object(container),
                )
            )
            seen |= set(objects)
        except ValueError as e:
            print(f"Skipping object {container} due to problems: {e}")

    if seen != objects:
        print(f"Missing containers: {all_objects - seen}")

    return tasks


# Assistant:
# 1. We are near the countertop.
# 2. We see a multitude of objects on the countertop, among them a toaster, a butter knife, a laptop, and a mug.
# 3. We are now holding the laptop, since it is at the bottom of the screen.
# 4. We are still holding the laptop. We are now walking away from the countertop.
# 5. We are now a bit further away from the countertop. We are still holding the laptop.


def prompt_container(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us putting down a {object} somewhere. Your eventual goal will be to recognize the place or container we put the object in (or on), but you shouldn't lock-in to one answer too early. Instead, try to describe the place or container as accurately as possible separately for each frame, refining your answer as you see more frames.
""".strip()


def example_container(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- You are bound not to recognize some objects correctly, which might hurt your peformance in downstream tasks. Instead of running with your first guess, try to list a few alternatives for an object if you're unsure what it is.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are in the middle of the room. We do not see any {object} yet.
2. The location has changed. We see something that might be the {object}. It seems to be in a pot which itself is on a sofa.
2. The scene has not changed much since the last frame. We are still near the sofa.
Considering the descriptions above, it is likely that the {object} was picked up from a pot, or possibly from a sofa.
4. Not much has changed, we are still near the sofa, and we still hold the {object},
5. We are now a bit further away from the sofa. We are still holding the {object}.
""".strip()


def container_tasks(trajectories: list[Trajectory]) -> list[Task]:
    all_containers = set(plausible_action_objects["PutObject"][1])
    container_trajectories = get_clipped_trajectories(trajectories, "PutObject")
    seen = set()
    tasks = []
    for object, containers, g in group_trajectories(
        container_trajectories, object1=True
    ):
        assert isinstance(object, str) and isinstance(containers, list)
        if seen == all_containers:
            break
        obj_trajectories = [
            t for container, t in zip(containers, g) if container not in seen
        ]
        if not obj_trajectories:
            continue
        if not object:
            object = "somewhere"
        try:
            tasks.append(
                Task(
                    prefix="foundation/containers",
                    name=f"place_{object.replace(' ', '_')}",
                    trajectories=obj_trajectories,
                    prompt_gpt=prompt_container(object),
                    example_gpt=example_container(object),
                )
            )
            seen |= set(containers)
        except ValueError as e:
            print(f"Skipping object {object} due to problems: {e}")

    if seen != all_containers:
        print(f"Missing containers: {all_containers - seen}")

    return tasks


def prompt_cleaning(object):
    return f"""
The frames will depict us carrying a {object} to a sink and then possibly cleaning it under running water. Your eventual goal will be to discern whether all the steps required for cleaning the object actually happened, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the objects we're holding, and from that deduce what actions have likely been performed since the last frame.
""".strip()


def example_cleaning(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- Cleaning an object involves holding it, putting it in a sink, running the water over it, stopping the water, and picking the object back up. Proper cleaning needs to have all these parts

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. There is a {object} at the bottom of the screen, which suggests we are holding it.
2. We are now near a sink. The {object} is still at the bottom of the screen, so we are definitely holding it.
3. The {object} now lies in the sink, along with other objects. The water is not running yet.
4. The {object} is still in the sink. The water is still not running.
5. We are holding the {object} again, as depicted by it being at the bottom of the screen again. Although it was in the sink, we didn't run water over it, which means we did not clean it.
"""


def clean_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    cleaning_trajectories = get_clipped_trajectories(trajectories, "CleanObject")
    groups = group_trajectories(cleaning_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()

        # Go to, Go away
        for t in g:
            idx = [a.action for a in t.actions].index("CleanObject")
            alt_actions = deepcopy(t.actions)
            alt_t = t.with_modified_actions(alt_actions[:idx] + alt_actions[idx + 1 :])
            alt_t._description = f"We go to the {container}, holding {object} in hand, we don't put it in the {container}"
            task_trajectories.append(alt_t)

        # Go to, Put, Pick, Go away
        for t in g:
            alt_t = t.filter_low_level(lambda lla: "ToggleObject" not in lla.action)
            alt_t._description = f"We put the {object} in the {container} and pick it back up without running water over it"
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/clean",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_cleaning(object),
                example_gpt=example_cleaning(object),
            )
        )
    return tasks


def prompt_heating(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us carrying a {object} to a microwave and then possibly heating it there. Your eventual goal will be to discern whether all the steps required for heating the object actually happened, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the objects we're holding, and what actions have likely been performed since the last frame.
""".strip()


def example_heating(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- Heating an object involves holding it, opening a microwave, putting it in a microwave, closing the microwave, turning the microwave on, turning it off, opening it, and picking the object back up. Proper heating needs to have all these parts.
- When the microwave is turned on, it lights up

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. There is a {object} at the bottom of the screen, which suggests we are holding it.
2. We are now near a microwave. The {object} is still at the bottom of the screen, so we are definitely holding it.
3. The {object} now lies in the microwave. The microwave is open and thus not running, so we are not heating the {object} up yet.
4. The {object} is still in the microwave. The microwave is still open.
5. We are holding the {object} again, as depicted by it being at the bottom of the screen again. Although it was in the microwave, we didn't close the microwave and turn it on, which means we did not clean it.
"""


def heat_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    heating_trajectories = get_clipped_trajectories(trajectories, "HeatObject")
    groups = group_trajectories(heating_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()

        sequence = [
            "OpenObject",
            "PutObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
            "OpenObject",
            "PickupObject",
            "CloseObject",
        ]

        # Go to, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("HeatObject")
            alt_t = t.with_modified_actions(alt_actions[:idx] + alt_actions[idx + 1 :])
            alt_t._description = f"We go to the {container} and then immediately leave without putting the {object} in"
            task_trajectories.append(alt_t)

        # Go to, Open, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("HeatObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:1] + llas[7:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container} and immediately close it again, without putting the {object} in"
            task_trajectories.append(alt_t)

        # Go to, Open, Put, Pick up, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("HeatObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:2] + llas[6:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container}, put the {object} in, then immediately pick it back up without heating it"
            task_trajectories.append(alt_t)

        # Go to, Open, Put, Close, Open, Pick up, Close, Go away
        for t in g:
            alt_t = t.filter_low_level(lambda lla: "ToggleObject" not in lla.action)
            alt_t._description = f"We put the {object} in the {container} for a while, we do not turn the {container} on, then we pick up the {object} back again"
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/heat",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_heating(object),
                example_gpt=example_heating(object),
            )
        )
    return tasks


# Assistant:


def prompt_cooling(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us carrying a {object} to a fridge and then possibly cooling it by leaving it in a closed fridge for a while before picking it up again. Your eventual goal will be to discern whether all the steps required for cooling the object actually happened, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the objects we're holding, and what actions have likely been performed since the last frame.
""".strip()


def example_cooling(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for cleaning, heating, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- Cooling an object involves holding it, opening the fridge, putting it in a fridge, closing the fridge, opening the fridge, and picking the object back up. Proper cooling needs to have all these parts.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. There is {object} at the bottom of the screen, which suggests we are holding it.
2. We are now near a fridge. The {object} is still at the bottom of the screen, so we are definitely holding it.
3. The fridge is now open. We still hold the {object}.
4. The fridge is now closed. We still hold the {object}. This means we did not put the object in the fridge.
5. We are holding the {object}, still. It seems we just opened and closed the fridge without putting the {object} in, even for a moment. The cooling did not happen.
"""


def cool_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    cooling_trajectories = get_clipped_trajectories(trajectories, "CoolObject")
    groups = group_trajectories(cooling_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()

        sequence = [
            "OpenObject",
            "PutObject",
            "CloseObject",
            "OpenObject",
            "PickupObject",
            "CloseObject",
        ]

        # Go to, Open, Put, Immediately pick up, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("CoolObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:2] + llas[4:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container}, put in the {object} and immediately pick it back up without leaving it to cool in a closed {container}"
            task_trajectories.append(alt_t)

        # Go to, Open, Close, Go away
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("CoolObject")
            hla = alt_actions[idx]
            llas = hla.actions
            if [lla.action for lla in llas] != sequence:
                continue
            hla.actions = llas[:1] + llas[5:]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = f"We open the {container} and then immediately close it without putting the {object} in"
            task_trajectories.append(alt_t)

        # Go to, Go away
        for t in g:
            idx = [a.action for a in t.actions].index("CoolObject")
            alt_actions = t.actions[:idx] + t.actions[idx + 1 :]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = (
                f"We go to the {container} and then we leave, without even opening it"
            )
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/cool",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_cooling(object),
                example_gpt=example_cooling(object),
            )
        )
    return tasks


def prompt_toggle(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict a {object}, and your eventual goal will be to discern whether we turned it on or off during the video. You shouldn't lock-in to one answer too early, though — instead, for each frame, concisely describe the scene and the state of the {object} and what actions have likely been performed since the last frame.
""".strip()


def example_toggle(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for toggling, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- We might be carrying an object in the frames, but what we care about mostly is turning other objects on or off.
- The state of the {object} will change exactly once.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. We see a {object} in front of us.
2. The {object} doesn't seem to be turned on; there is a red light on the front that signals it is off.
3. Nothing has changed; the {object} is still off.
4. Now, the {object} is turned on. We can see this because it the light has turned on.
5. The {object} is still on.
"""


def toggle_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []

    toggling_trajectories = []
    for t in trajectories:
        toggling_indices = [
            (i, j)
            for i, hla in enumerate(t.actions)
            for j, lla in enumerate(hla.actions)
            if lla.action == "ToggleObjectOn"
        ]
        for hla_idx, lla_idx in toggling_indices:
            hl_actions = deepcopy(t.actions)
            hl_actions[hla_idx - 1].actions = hl_actions[hla_idx - 1].actions[-3:]
            beg = (
                hla_idx - 1
                if hla_idx > 0 and hl_actions[hla_idx - 1].action == "GotoLocation"
                else hla_idx
            )
            end = hla_idx + 1
            toggle_action = hl_actions[hla_idx]
            toggle_action.actions = [toggle_action.actions[lla_idx]]
            new_t = t.with_modified_actions(hl_actions[beg:end])
            new_t._description = f"We turn the {toggle_action.actions[0].object1} on"
            toggling_trajectories.append((hla_idx - beg, new_t))

    for g in utils.group_by(
        toggling_trajectories, lambda t: t[1].actions[t[0]].actions[0].object1
    ):
        object, g = g[0][1].actions[g[0][0]].actions[0].object1, [t for _, t in g]

        task_trajectories = g.copy() + [
            t.reverse(f"We turn the {object} off") for t in g
        ]

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/toggle",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_toggle(object),
                example_gpt=example_toggle(object),
            )
        )

    return tasks


def prompt_pick(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us handling a {object}, and your eventual goal will be to discern whether picked it up during the video, or started out holding it and put it down somewhere. You shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_pick(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for putting an object down, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- Sometimes it is hard to see where exactly we put the object. However, if we held it and we don't, we definitely had to put it somewhere.
- We will never pick up an object and put it down in the same frame. Only exactly one of these will happen.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near a countertop or a large table.
2. Now it seems we walked near a windowsill. We don't appear to have anything in hand. There is a {object} on the windowsill, along with a toaster, and a bowl or a plate.
3. We seem to be holding the {object} now, because it is at the bottom of the screen, almost as if it was lying on the floor.
4. We still hold the {object}. We are further from the windowsill now, possibly walking towards somewhere else.
5. We still hold the {object}.
"""


def pick_up_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    picking_trajectories = get_clipped_trajectories(trajectories, "PickupObject")
    putting_trajectories = get_clipped_trajectories(trajectories, "PutObject")
    groups = group_trajectories(
        picking_trajectories + putting_trajectories, object1=True, object2=True
    )
    for object, container, g in groups:
        if container is None or not set(
            a.action for t in g for a in t.actions
        ).issuperset({"PickupObject", "PutObject"}):
            continue
        assert isinstance(object, str) and isinstance(container, str)

        name = f"{object.replace(' ', '_')}_{container.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/pick_v_put",
                name=name,
                trajectories=g,
                prompt_gpt=prompt_pick(object),
                example_gpt=example_pick(object),
            )
        )

    return tasks


def slicing_prompt(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us carrying a knife and walking towards a {object}. Your eventual goal will be to discern whether we sliced the {object} at the end of the video or not, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_slicing(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for putting an object down, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- Sometimes we just walk to an object with a knife in hand without slicing the object. To see whether an object has been sliced, first find it in the frame, and then observe how it changes in the following frames. A sliced object will have small, barely noticeable lines on it.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. A butter knife is shown on the bottom of the screen, which means we are holding it, as expected.
2. We have walked near a windowsill, upon which there is a multitude of objects, including {object}. The knife is still at the bottom of the screen, so we are definitely holding it.
3. We are holding the knife still, but it is not clear if we are using it. Focusing on the {object}, it seems unchanged.
4. The {object} now has lines on it, which suggests we have sliced it. We still hold the knife, and the {object} is still in the same place.
5. We are still near the windowsill. We now see the {object} from a different angle, and although we now do not see the slicing lines, we did slice it according to the previous frame.
"""


def slice_tasks(trajectories: list[Trajectory]) -> list[Task]:
    slicing_trajectories = get_clipped_trajectories(trajectories, "SliceObject")
    tasks = []
    for object, _, g in group_trajectories(slicing_trajectories, object1=True):
        assert isinstance(object, str)
        task_trajectories = g.copy()
        for t in g:
            idx = [a.action for a in t.actions].index("SliceObject")
            if idx == 0:
                continue
            alt_t = t[:idx]
            alt_t._description = (
                f"We walk to the {object} with a knife in hand but don't slice it"
            )
            task_trajectories.append(alt_t)

        tasks.append(
            Task(
                prefix="foundation/slice",
                name=f"{object.replace(' ', '_')}",
                trajectories=task_trajectories,
                prompt_gpt=slicing_prompt(object),
                example_gpt=example_slicing(object),
            )
        )
    return tasks


def sliced_v_whole_prompt(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will depict us picking up a {object}, or possibly just a piece or slice of it. Your eventual goal will be to say whether the object was picked up whole, or if just a piece of it was picked, but you shouldn't lock-in to one answer too early. Instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_sliced_v_whole(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for picking an object up, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- We are sure that you will observe a {object} being picked up. If you can barely see it, maybe you just picked up a very thin slice that is angled weirdly.
- Sometimes we will pick up slices, sometimes wedges, but you can usually ignore this, as long as you can tell whether the object is whole or not.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are nearing a countertop. We see a {object} lying on the countertop. It is hard to see whether it is sliced or not when it's lying down like this.
2. We are now standing directly at the countertop.
3. We picked up the {object}. It seems to be a wedge instead of the whole thing.
4. We now changed our position again. We can now see the potato is just a wedge since we can see the inner texture.
5. We are now nearing a fridge, still holding the {object}, as signified by it being at the bottom of the screen.
"""


def sliced_v_whole_tasks(trajectories: list[Trajectory]) -> list[Task]:
    picking_trajectories = get_clipped_trajectories(trajectories, "PickupObject")
    groups = utils.group_by(
        picking_trajectories,
        lambda t: (t[1].actions[t[0]].object1.removeprefix("slice of ")),
    )
    tasks = []
    for g in groups:
        if len(set(t.actions[idx].object1 for idx, t in g)) == 1:
            # Skip, we are only interested with actions where the object is both sliced and non-sliced
            continue
        object = g[0][1].actions[g[0][0]].object1.removeprefix("slice of ")
        # container = g[0][1].actions[g[0][0]].object2 or "somewhere"
        task_trajectories = []
        for idx, t in g:
            t._description = f"We pick up a {t.actions[idx].object1}"
            task_trajectories.append(t)
        tasks.append(
            Task(
                prefix="foundation/sliced_v_whole",
                name=object.replace(" ", "_"),
                trajectories=task_trajectories,
                prompt_gpt=sliced_v_whole_prompt(object),
                example_gpt=example_sliced_v_whole(object),
            )
        )
    return tasks


def prompt_on_v_off(object):
    return f"""
Your task is to describe what you see in each frame, separately, in a list. The frames will standing in front of or walking towards a {object}. Your eventual goal will be to say whether the {object} has been on or off in the video. The on/off state of the object won't change during the video. You shouldn't lock-in to one answer too early — instead, for each frame, concisely describe the scene, the state of the {object}, and what actions have likely been performed since the last frame.
""".strip()


def example_on_v_off(object):
    return f"""
A few tips and hints to help you get started:
- 'Holding an object' is depicted as the object being at the bottom of the screen, without any visible hands.
- Similarly, no hands are shown for putting an object down, or any other action.
- The objects almost never lie on the floor. If the object is at the bottom of the screen and looks like it is lying on the floor, we are actually just holding it.
- Sometimes the {object} will be barely visible in the frame. Still, you should be able to discern whether it is shining (and thus on) or not.

Example with the tips applied:

(note that this example is for five frames for illustration purposes, but you should work with as many frames as you are given)

Input: [5 frames]

Frame-by-frame description:
1. We are near the countertop. We see a {object} in front of us.
2. The {object} doesn't seem to be turned on. There is no light coming from it.
3. Nothing has changed; the {object} is still off.
4. Again, similar frame, the {object} is still off, as we would expect.
5. For the entire length of the video, the {object} has been off.
"""


def on_v_off_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []

    toggling_trajectories = []
    for t in trajectories:
        toggling_indices = [
            (i, j)
            for i, hla in enumerate(t.actions)
            for j, lla in enumerate(hla.actions)
            if lla.action == "ToggleObjectOn"
        ]
        for hla_idx, lla_idx in toggling_indices:
            hl_actions = deepcopy(t.actions)
            hl_actions[hla_idx - 1].actions = hl_actions[hla_idx - 1].actions[-5:]
            if len(hl_actions) > hla_idx + 1:
                hl_actions[hla_idx + 1].actions = hl_actions[hla_idx + 1].actions[:5]
            beg = (
                hla_idx - 1
                if hla_idx > 0 and hl_actions[hla_idx - 1].action == "GotoLocation"
                else hla_idx
            )
            end = (
                hla_idx + 2
                if hla_idx + 1 < len(hl_actions)
                and hl_actions[hla_idx + 1].action == "GotoLocation"
                else hla_idx + 1
            )
            toggle_action = hl_actions[hla_idx]
            toggle_action.actions = [toggle_action.actions[lla_idx]]
            new_t = t.with_modified_actions(hl_actions[beg:end])
            toggling_trajectories.append((hla_idx - beg, new_t))

    for g in utils.group_by(
        toggling_trajectories, lambda t: t[1].actions[t[0]].actions[0].object1
    ):
        object = g[0][1].actions[g[0][0]].actions[0].object1

        task_trajectories = []
        for idx, t in g:
            t_on, t_off = deepcopy(t), deepcopy(t)
            t_on.actions = [t_on.actions[idx]]
            t_on.actions[0].actions[0].images = t.actions[idx].actions[0].images[-3:]
            t_on._description = f"The {object} is turned on"
            t_off.actions[idx].actions[0].images = t.actions[idx].actions[0].images[:3]
            t_off._description = f"The {object} is turned off"
            task_trajectories += [t_on, t_off]

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                prefix="foundation/on_v_off",
                name=name,
                trajectories=task_trajectories,
                prompt_gpt=prompt_on_v_off(object),
                example_gpt=example_on_v_off(object),
            )
        )

    return tasks


def actions_to_blocks(actions: list[HighLevelAction]) -> list[list[HighLevelAction]]:
    grouped_actions, group = [], []
    for a in actions:
        if group and group[-1].action != "GotoLocation":
            grouped_actions.append(group)
            group = []
        group.append(a)
    if group:
        grouped_actions.append(group)
    return grouped_actions


def from_action_blocks(blocks: list[list[HighLevelAction]]) -> list[HighLevelAction]:
    return [a for block in blocks for a in block]


def equivalent_trajectory_present(
    trajectory: Trajectory, trajectories: list[Trajectory]
):
    actions = [a for a in trajectory.actions if a.action != "GotoLocation"]
    for t in trajectories:
        alt_actions = [a for a in t.actions if a.action != "GotoLocation"]
        if all(a1 == a2 for a1, a2 in zip(actions, alt_actions)):
            return True
    return False


def trajectories_with_block_prefix(
    current_blocks: list[list[HighLevelAction]],
    target_len: int,
    not_equal_to: list[HighLevelAction],
    scene: str,
    all_blocks: dict[tuple[str, str | None], list[list[HighLevelAction]]],
    all_goto: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ],
):
    if len(current_blocks) == target_len:
        yield current_blocks
        return

    next_blocks = []

    if current_blocks:
        last_action = current_blocks[-1][-1]
        location, hand_content = last_action.end_location, last_action.end_hand_content
        for b in all_blocks.get((scene, hand_content), []):
            a = b[0] if b[0].action != "GotoLocation" else b[1]
            # The first added step in this trajectory has to be different
            # from the other first added steps in the other trajectories
            if a in not_equal_to:
                continue

            # If literally the same action is already in the trajectory, skip
            if any(a is other_a for block in current_blocks for other_a in block):
                continue

            if b[0].beg_location == location:
                next_blocks.append(b)
            else:
                gotos = all_goto.get(scene, {}).get(
                    (location, a.beg_location, hand_content), []
                )
                for goto in gotos:
                    next_blocks.append([goto] + b)
    else:
        for (s, _), l in all_blocks.items():
            if s == scene:
                for b in l:
                    a = b[0] if b[0].action != "GotoLocation" else b[1]
                    if a in not_equal_to:
                        continue
                    next_blocks.append(b)

    for b in next_blocks:
        new_trajectory = current_blocks + [b]
        yield from trajectories_with_block_prefix(
            current_blocks=new_trajectory,
            target_len=target_len,
            # The next step can be anything now, just the first step had to be different
            not_equal_to=[],
            scene=scene,
            all_blocks=all_blocks,
            all_goto=all_goto,
        )


def compute_blocks(
    trajectories: list[Trajectory],
) -> dict[tuple[str, str | None], list[list[HighLevelAction]]]:
    all_blocks: dict[tuple[str, str | None], list[list[HighLevelAction]]] = {}
    for trajectory in trajectories:
        for g in actions_to_blocks(trajectory.actions):
            scene, hand_content = g[0].scene, g[0].beg_hand_content
            all_blocks.setdefault((scene, hand_content), []).append(g)
    return all_blocks


def compute_gotos(
    trajectories: list[Trajectory],
) -> dict[str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]]:
    goto_actions: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ] = {}
    for trajectory in trajectories:
        for a in [a for a in trajectory.actions if a.action == "GotoLocation"]:
            scene_gotos = goto_actions.setdefault(a.scene, {})
            matching_actions = scene_gotos.setdefault(
                (a.beg_location, a.end_location, a.beg_hand_content), []
            )
            matching_actions.append(a)
    return goto_actions


def block_to_action(block: list[HighLevelAction]) -> HighLevelAction:
    return block[0] if block[0].action != "GotoLocation" else block[1]


def remixed_task(
    seed: Trajectory,
    videos_on_prefix_0: int,
    videos_per_prefix: int,
    trajectories: list[Trajectory],
) -> Task:
    all_blocks = compute_blocks(trajectories)
    all_gotos = compute_gotos(trajectories)

    seed_blocks = actions_to_blocks(seed.actions)
    target_len = len(seed_blocks)
    trajectories_by_prefix_len = {target_len: [seed]}
    for prefix_len, seed_block in enumerate(seed_blocks):
        if len(seed_block) == 1 and seed_block[0].action == "GotoLocation":
            continue

        candidates = trajectories_with_block_prefix(
            seed_blocks[:prefix_len].copy(),
            target_len,
            # The upcoming action should be different from all the other first actions
            # among the trajectories with the same prefix length (and seed trajectory)
            [
                block_to_action(actions_to_blocks(t.actions)[prefix_len])
                for t in trajectories_by_prefix_len.get(prefix_len, []) + [seed]
            ],
            seed.actions[0].scene,
            all_blocks,
            all_gotos,
        )

        num_classes_limit = videos_on_prefix_0 if prefix_len == 0 else videos_per_prefix
        num_classes = 0
        num_iters = 0
        for trajectory in candidates:
            num_iters += 1
            if num_classes == num_classes_limit:
                break
            new_t = seed.with_modified_actions(from_action_blocks(trajectory))
            if not equivalent_trajectory_present(
                new_t, [t for l in trajectories_by_prefix_len.values() for t in l]
            ):
                trajectories_by_prefix_len.setdefault(prefix_len, []).append(new_t)
                num_classes += 1

    return Task(
        prefix=f"level_{target_len}/remix",
        name=str(uuid.uuid4()),
        trajectories=[t for l in trajectories_by_prefix_len.values() for t in l],
        metadata={
            "prefix_lens": [
                {"shared_prefix": prefix_len, "description": t.description}
                for prefix_len, trajectories in trajectories_by_prefix_len.items()
                for t in trajectories
            ]
        },
        prompt_gpt=HIGH_LEVEL_TASK_PROMPT,
        example_gpt=HIGH_LEVEL_TASK_EXAMPLE,
    )


def try_make_consistent(
    actions: list[HighLevelAction],
    all_goto: dict[
        str, dict[tuple[str | None, str | None, str | None], list[HighLevelAction]]
    ],
) -> list[HighLevelAction] | None:
    result = [actions[0]]
    for prev, next in zip(actions, actions[1:]):
        if prev.end_hand_content != next.beg_hand_content:
            return None

        if prev.end_location != next.beg_location:
            gotos = all_goto.get(prev.scene, {}).get(
                (prev.end_location, next.beg_location, prev.end_hand_content), []
            )
            if not gotos:
                return None
            result += [gotos[0], next]
        else:
            result.append(next)
    return result


def slicing_consistent(actions: list[HighLevelAction]) -> bool:
    for slicing_i in [i for i, a in enumerate(actions) if a.action == "SliceObject"]:
        sliced_object = actions[slicing_i].object1
        whole_before = all(
            i <= slicing_i
            for i, a in enumerate(actions)
            if (sliced_object in a.description and "slice" not in a.description)
        )
        sliced_after = all(
            i >= slicing_i
            for i, a in enumerate(actions)
            if ("slice" in a.description and sliced_object in a.description)
        )
        if not (whole_before and sliced_after):
            return False
    return True


def permutation_task(
    seed: Trajectory,
    num_classes: int,
    trajectories: list[Trajectory],
) -> Task:
    goto_actions = compute_gotos(trajectories)
    important_actions = [a for a in seed.actions if a.action != "GotoLocation"]

    consistent_trajectories, inconsistent_trajectories = [seed], []
    for perm in itertools.permutations(important_actions):
        if len(consistent_trajectories) == num_classes:
            break

        new_actions = try_make_consistent(list(perm), goto_actions)
        if new_actions is None:
            continue

        new_trajectory = seed.with_modified_actions(new_actions)

        if not equivalent_trajectory_present(
            new_trajectory, consistent_trajectories + inconsistent_trajectories
        ):
            if slicing_consistent(list(perm)):
                # if "we slice" not in perm[0].description:
                #     for a in perm:
                #         print(a.description)
                #     print()
                consistent_trajectories.append(new_trajectory)
            else:
                inconsistent_trajectories.append(new_trajectory)

    task_trajectories = (consistent_trajectories + inconsistent_trajectories)[
        :num_classes
    ]

    return Task(
        prefix=f"level_{len(important_actions)}/permutation",
        name=str(uuid.uuid4()),
        trajectories=task_trajectories,
        prompt_gpt=HIGH_LEVEL_TASK_PROMPT,
        example_gpt=HIGH_LEVEL_TASK_EXAMPLE,
        metadata={
            "is_consistent_wrt_slicing": len(consistent_trajectories) == num_classes,
        },
    )
