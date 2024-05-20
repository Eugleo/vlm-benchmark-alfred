import itertools
import json
import random
import uuid
from collections import Counter
from copy import deepcopy

import yaml

from alfred import utils
from alfred.object import plausible_action_objects
from alfred.trajectory import HighLevelAction, Trajectory

PROMPT_GPT = """
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Focus on our location, the objects we're handling, and what actions are (likely) being performed. The ways in which you can interact with objects include cleaning them, heating them up, cooling them down, picking them up, and putting them somewhere.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it.

## EXAMPLE
Input: [five frames]

Assistant:
1. We are near the countertop.
2. Now it seems we walked near a windowsill. We don't appear to have anything in hand. There is a butter knife on the windowsill.
3. We seem to be holding the butter knife now, beacuse it is at the bottom of the screen, almost as if it was lying on the floor. We are near a microwave.
4. We now see the microwave is turned on. We are probaly heating up the butter knife, since we do not have it in our hand anymore.
5. We have pulled out the butter knife from the microwave. We are now near the windowsill again."""[
    1:
]


class Task:
    prefix: str
    name: str
    prompt_gpt: str
    trajectories: list[Trajectory]
    descriptions: list[str]

    def __init__(
        self,
        prefix,
        name,
        trajectories,
        prompt_gpt: str = PROMPT_GPT,
        descriptions=None,
    ):
        self.prefix = prefix
        self.name = name
        self.trajectories = trajectories
        self.prompt_gpt = prompt_gpt
        if descriptions is None:
            descriptions = list(set(t.description for t in trajectories))
        self.descriptions = descriptions
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

    def write(self, output_dir, max_videos=5, min_frames=32) -> str:
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
                    "concepts": self._used_concepts,
                },
                f,
            )

        videos = []
        video_counter = Counter()
        # Task videos
        # TODO Could be made more efficient by only having one copy of each video
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

    @staticmethod
    def create_permuted(trajectories: list[Trajectory], label_count: int):
        master = trajectories[0]
        permuted_trajectories, seen = deepcopy(trajectories), {tuple(master.actions)}
        label_count -= 1

        for permutation in itertools.permutations(range(len(master))):
            if label_count <= 0:
                break

            actions = tuple(master.actions[i] for i in permutation)
            if actions in seen:
                continue
            seen.add(actions)
            label_count -= 1

            for trajectory in trajectories:
                actions = [trajectory.actions[i] for i in permutation]
                permuted_trajectories.append(trajectory.with_modified_actions(actions))

        return Task(
            prefix=f"level_{len(master)}/permuted",
            name=str(uuid.uuid4()),
            trajectories=permuted_trajectories,
        )

    @staticmethod
    def create_substituted(
        trajectories: list[Trajectory],
        action_library: dict[str, list[HighLevelAction]],
        label_count: int,
    ):
        master = trajectories[0]
        substituted_trajectories, seen = deepcopy(trajectories), {}

        for _ in range(label_count - 1):
            index = random.randint(0, len(master) - 1)
            master_action = master.actions[index]
            alternative_action = random.choice(
                [
                    a
                    for a in action_library[master_action.action]
                    if a not in seen.get(index, {})
                ]
            )
            seen.setdefault(index, set()).add(alternative_action)
            substituted_trajectories += [
                t.substitute(index, alternative_action) for t in trajectories
            ]

        return Task(
            prefix=f"level_{len(master)}/substituted",
            name=str(uuid.uuid4()),
            trajectories=substituted_trajectories,
        )


def write_config(tasks, output_dir):
    # Config
    with open(output_dir / "configs" / "alfred.yaml", "w") as f:
        config = {
            "tasks": tasks,
            "models": [
                {"kind": "encoder", "encoder": "s3d", "heads": [{"kind": "cosine"}]},
                # {"kind": "gpt", "n_frames": 5},
                {"kind": "encoder", "encoder": "viclip", "heads": [{"kind": "cosine"}]},
                # {
                #     "kind": "encoder",
                #     "encoder": "clip",
                #     "heads": [{"kind": "cosine"}],
                #     "hf_model": "ViT-bigG-14/laion2b_s39b_b160k",
                #     "n_frames": 8,
                # },
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
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us or putting down an object into/on a {container}. Your goal is to discern the the object we are handling.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera.

## EXAMPLE
Input: [five frames]

Assistant:
1. We are near the {container}.
2. We see a multitude of objects on the {container}, among them a toaster, a butter knife, a laptop, and a mug.
3. We are now holding the laptop, since it is at the bottom of the screen.
4. We are still holding the laptop. We are now walking away from the {container}.
5. We are now a bit further away from the {container}. We are still holding the laptop."""[
        1:
    ]


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
                    "foundation/objects",
                    f"pick_from_{name}",
                    container_trajectories,
                    prompt_gpt=prompt_object(container),
                )
            )
            seen |= set(objects)
        except ValueError as e:
            print(f"Skipping object {container} due to problems: {e}")

    if seen != objects:
        print(f"Missing containers: {all_objects - seen}")

    return tasks


def prompt_container(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us or putting a {object} somewhere. Your goal is to discern the container we are putting it in/on.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera.

## EXAMPLE
Input: [five frames]

Assistant:
Assistant:
1. We are near the countertop.
2. We see a multitude of objects on the countertop, among them a toaster, a butter knife, a laptop, and a mug.
3. We are now holding the laptop, since it is at the bottom of the screen.
4. We are still holding the laptop. We are now walking away from the countertop.
5. We are now a bit further away from the countertop. We are still holding the laptop."""[
        1:
    ]


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
                    "foundation/containers",
                    f"place_{object.replace(' ', '_')}",
                    obj_trajectories,
                    prompt_gpt=prompt_container(object),
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
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us carrying a {object} to a sink and then possibly cleaning it under running water. Your goal is to discern whether the cleaning actually happened — i.e. whether the water was running — or whether we just moved the {object} around.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it. Similarly, cleaning an object is depicted by the object being in a sink, submerged in water.

## EXAMPLE FOR A SIMILAR TASK (heating)
Input: [five frames]

Assistant:
1. We are near the countertop. There is keychain at the bottom of the screen, which suggests we are holding it.
2. We are now near a sink. The keychain is still at the bottom of the screen, so we are definitely holding it.
3. The keychain now lies in the sink. The water is not running, so we are not cleaning it.
4. The keychain is still in the sink. The water is not running, so we are not cleaning it.
5. We are holding the keychain again. We did not clean it."""[1:]


def clean_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    cleaning_trajectories = get_clipped_trajectories(trajectories, "CleanObject")
    groups = group_trajectories(cleaning_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()
        for t in g:
            alt_t = t.filter_low_level(lambda lla: "ToggleObject" not in lla.action)
            alt_t._description = f"We put the {object} in the {container} and pick it back up without running water over it"
            task_trajectories.append(alt_t)

        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                "foundation/clean",
                name,
                task_trajectories,
                prompt_gpt=prompt_cleaning(object),
            )
        )
    return tasks


def prompt_heating(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us carrying a {object} to a microwave and then possibly heating it up there. Your goal is to discern whether the microwave has actually been turned on, or whether we just moved the {object} around without heating it up.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it.

## EXAMPLE (heating)
Input: [five frames]

Assistant:
1. We are near the countertop. There is keychain at the bottom of the screen, which suggests we are holding it.
2. We are now near a microwave. The keychain is still at the bottom of the screen, so we are definitely holding it.
3. The keychain now lies in the microwave. The microwave is not running, so we are not heating the keychain up.
4. The keychain is still in the microwave. The microwave is still not running.
5. We are holding the keychain again. We did not heat it up."""[1:]


def heat_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    heating_trajectories = get_clipped_trajectories(trajectories, "HeatObject")
    groups = group_trajectories(heating_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()
        for t in g:
            alt_t = t.filter_low_level(lambda lla: "ToggleObject" not in lla.action)
            alt_t._description = f"We put the {object} in the {container} and pick it back up without heating it"
            task_trajectories.append(alt_t)
        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                "foundation/heat",
                name,
                task_trajectories,
                prompt_gpt=prompt_heating(object),
            )
        )
    return tasks


def prompt_cooling(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us carrying a {object} to a fridge and then possibly cooling it down there for a while. Your goal is to discern whether the {object} has actually been in the fridge for a while, or whether we just moved it around without heating it up.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it.

## EXAMPLE (heating)
Input: [five frames]

Assistant:
1. We are near the countertop. There is keychain at the bottom of the screen, which suggests we are holding it.
2. We are now near a fridge. The keychain is still at the bottom of the screen, so we are definitely holding it.
3. The fridge is now open. We still hold the keychain.
4. The fridge is now closed. We still hold the keychain.
5. We are holding the keychain, still. It seems we just opened and closed the fridge without putting the keychain in, even for a moment."""[
        1:
    ]


def cool_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    cooling_trajectories = get_clipped_trajectories(trajectories, "CoolObject")
    groups = group_trajectories(cooling_trajectories, object1=True, object2=True)
    for object, container, g in groups:
        assert isinstance(object, str) and isinstance(container, str)
        task_trajectories = g.copy()
        for t in g:
            alt_actions = deepcopy(t.actions)
            idx = [a.action for a in t.actions].index("CoolObject")
            hla = alt_actions[idx]
            llas = hla.actions
            put_idx = [lla.action for lla in llas].index("PutObject")
            # Drop the whole sequence where we put the object into the fridge
            # Then close the fridge etc
            hla.actions = llas[:put_idx] + llas[put_idx + 4 :]
            alt_t = t.with_modified_actions(alt_actions)
            alt_t._description = (
                f"We open the {container} and then close it without putting anything in"
            )
            task_trajectories.append(alt_t)
        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                "foundation/cool",
                name,
                task_trajectories,
                prompt_gpt=prompt_cooling(object),
            )
        )
    return tasks


def prompt_toggle(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will depict us standing in front of a {object} and your goal is to see whether we turned it on or off.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it. Simialrly, whether an object has changed state is depicted only by a (sudden) change in the object's appearance.

## EXAMPLE (blender)
Input: [five frames]

Assistant:
1. We are near the countertop. We see a blender in front of us.
2. The blender doesn't seem to be turned on; there is a red light on the front that signals it is off.
3. Nothing has changed; the blender is still off.
4. Now, the blender is turned on. We can see the blades are in motion and the red light has turned green.
5. The light is still green, the blender is still on."""[1:]


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
                "foundation/toggle",
                name,
                task_trajectories,
                prompt_gpt=prompt_toggle(object),
            )
        )

    return tasks


def prompt_pick(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will depict us either picking up a {object}, or putting it down somewhere. Your goal is to discern whether we picked up the {object} or put it down.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera. Similarly, putting an object down is depicted by the object being placed on a surface, there is no intermediate movement.

## EXAMPLE (blender)
Input: [five frames]

Assistant:
1. We are near the countertop.
2. Now it seems we walked near a windowsill. We don't appear to have anything in hand. There is a butter knife on the windowsill.
3. We seem to be holding the butter knife now, beacuse it is at the bottom of the screen, almost as if it was lying on the floor.
4. We still hold the butter knife. We are further from the windowsill now, possibly walking towards somewhere else.
5. We still hold the knife."""[1:]


def pick_up_tasks(trajectories: list[Trajectory]) -> list[Task]:
    tasks = []
    picking_trajectories = get_clipped_trajectories(trajectories, "PickupObject")
    for object, _, g in group_trajectories(picking_trajectories, object1=True):
        assert isinstance(object, str)
        task_trajectories = g.copy() + [
            t.reverse(f"We put the {object} down") for t in g
        ]
        name = f"{object.replace(' ', '_')}"
        tasks.append(
            Task(
                "foundation/pick_v_put",
                name,
                task_trajectories,
                prompt_gpt=prompt_pick(object),
            )
        )

    return tasks


def slicing_prompt(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us carrying a knife and walking towards a {object}. Your goal is to discern whether we sliced the {object} at the end of the video or not. Focus very closely on the object and compare its state among the frames.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it. No movement will be shown either, the slicing is an instantaneous event that only slighty changes the object's appearance.

## EXAMPLE (grating)
Input: [five frames]

Assistant:
1. We are near the countertop. A grater is shown on the bottom of the screen, which probably means we are holding it.
2. We have walked near a windowsill. The grater is still at the bottom of the screen, so we are definitely holding it.
3. On the windowsill, we see a cucumber. We are holding the grater, but it is not clear if we are using it.
4. The cucumber now has lines on it, which suggests we have grated it.
5. We are still near the windowsill. The cucumber still has the grated appearance."""[
        1:
    ]


def slice_tasks(trajectories: list[Trajectory]) -> list[Task]:
    slicing_trajectories = get_clipped_trajectories(trajectories, "SliceObject")
    tasks = []
    for object, _, g in group_trajectories(slicing_trajectories, object1=True):
        assert isinstance(object, str)
        task_trajectories = g.copy()
        for t in g:
            idx = [a.action for a in t.actions].index("SliceObject")
            alt_t = deepcopy(t[: idx + 1])
            alt_t._description = (
                f"We walk to the {object} with a knife in hand but don't slice it"
            )
            task_trajectories.append(alt_t)

        tasks.append(
            Task(
                "foundation/slice",
                f"{object.replace(' ', '_')}",
                task_trajectories,
                prompt_gpt=slicing_prompt(object),
            )
        )
    return tasks


def sliced_v_whole_prompt(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will show us picking up a {object}. Your goal is to discern whether it is a whole {object} that we pick up or just a slice of it.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it.

## EXAMPLE (grating)
Input: [five frames]

Assistant:
1. We are nearing a countertop. We see a potato lying on the countertop.
2. We are now standing directly at the countertop. It is hard too see whether the potato is sliced or not when it just lies there.
3. We picked up the potato. It seems to be a wedge instead of the whole thing.
4. It is definitely just a wedge. We are turning away from the countertop now.
5. We are now nearing a fridge, still holding the potato, as signified by it being at the bottom of the screen."""[
        1:
    ]


def sliced_v_whole_tasks(trajectories: list[Trajectory]) -> list[Task]:
    picking_trajectories = get_clipped_trajectories(trajectories, "PickupObject")
    groups = utils.group_by(
        picking_trajectories,
        lambda t: (
            t[1].actions[t[0]].object1.removeprefix("slice of "),
            t[1].actions[t[0]].object2,
        ),
    )
    tasks = []
    for g in groups:
        object = g[0][1].actions[g[0][0]].object1
        if "slice" in object:
            print(g)
        if len(set(t.actions[idx].object1 for idx, t in g)) == 1:
            # Skip, we are only interested with actions where the object is both sliced and non-sliced
            continue
        tasks.append(
            Task(
                "foundation/sliced_v_whole",
                f"{object.replace(' ', '_')}",
                [t for _, t in g],
                prompt_gpt=sliced_v_whole_prompt(object),
            )
        )
    return tasks


def prompt_on_v_off(object):
    return f"""
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

# FIRST TASK

Your first task is to describe what you see in each frame. Specifically, the video will depict us standing in front of or walking towards a {object} and your goal is to see whether the {object} is turned on or off. It won't change state during the video.

Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it.

## EXAMPLE (blender)
Input: [five frames]

Assistant:
1. We are near the countertop. We see a blender in front of us.
2. The blender doesn't seem to be turned on; there is a red light on the front that signals it is off.
3. Nothing has changed; the blender is still off.
4. The red light is still on.
5. For the entire length of the video, the blender has been off"""[1:]


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
                "foundation/on_v_off",
                name,
                task_trajectories,
                prompt_gpt=prompt_on_v_off(object),
            )
        )

    return tasks
