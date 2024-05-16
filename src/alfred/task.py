import itertools
import json
import random
import uuid
from copy import deepcopy

import yaml

from alfred.trajectory import HighLevelAction, Trajectory, permute_trajectories


class Task:
    prefix: str
    name: str
    trajectories: list[Trajectory]
    descriptions: list[str]

    def __init__(self, prefix, name, trajectories, descriptions=None):
        self.prefix = prefix
        self.name = name
        self.trajectories = trajectories
        if descriptions is None:
            descriptions = [t.description for t in trajectories]
        self.descriptions = descriptions

    def write(self, output_dir) -> str:
        tasks_dir = output_dir / "tasks" / "alfred" / self.prefix
        tasks_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = output_dir / "videos" / "alfred" / self.prefix / self.name
        videos_dir.mkdir(parents=True, exist_ok=True)

        description_to_label = {
            d: f"label_{i}" for i, d in enumerate(self.descriptions)
        }
        label_to_description = {v: k for k, v in description_to_label.items()}

        # Task definition
        with open(tasks_dir / f"{self.name}.yaml", "w") as f:
            yaml.dump(
                {"label_prompts": label_to_description, "prompt_gpt": PROMPT_GPT}, f
            )

        videos = []
        # Task videos
        # TODO Could be made more efficient by only having one copy of each video
        for trajectory in self.trajectories:
            video_name = f"{trajectory.trial_id}_{uuid.uuid4()}.mp4"
            trajectory.video.write_videofile(str(videos_dir / video_name), logger=None)
            label = description_to_label[trajectory.description]
            videos.append({"path": f"{dir}/{self.name}/{video_name}", "label": label})

        # Task data
        with open(tasks_dir / f"{self.name}_data.json", "w") as f:
            json.dump(videos, f)

        return f"{self.prefix}/{self.name}"

    @staticmethod
    def create_permuted(trajectories: list[Trajectory], label_count: int):
        master = trajectories[0]
        permuted_trajectories, seen = deepcopy(trajectories), {tuple(master.actions)}

        for permutation in itertools.permutations(range(len(master))):
            if label_count <= 0:
                break

            master_actions = tuple(master.actions[i] for i in permutation)
            if master_actions in seen:
                continue
            seen.add(master_actions)
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
        action_library: list[HighLevelAction],
        label_count: int,
    ):
        master = trajectories[0]
        substituted_trajectories, seen = deepcopy(trajectories), {tuple(master.actions)}

        while label_count > 0:
            index = random.randint(0, len(master) - 1)
            if master_actions in seen:
                continue
            seen.add(master_actions)
            label_count -= 1

            for trajectory in trajectories:
                substituted_trajectories.append(trajectory)

        return Task(
            prefix=f"level_{len(master)}/substituted",
            name=str(uuid.uuid4()),
            trajectories=substituted_trajectories,
        )


PROMPT_GPT = """
You will be given five frames from a first-person video taken in a 3D model of a small house. The frames are given to you in chronological order.

Your task is to describe what you see in each frame. Focus on our location, the objects we're handling, and what actions are (likely) being performed. The ways in which you can interact with objects include cleaning them, heating them up, cooling them down, picking them up, and putting them somewhere.

IMPORTANT: Remember that this is a model, so the objects don't look exactly as they would in real life. For example, no hands will be shown in the frames; instead, 'holding' an object is depicted by the object being at the bottom of the frame, close to the camera, and moving with it.

# EXAMPLE
Input: [five frames]

Assistant:
1. We are near the countertop.
2. Now it seems we walked near a windowsill. We don't appear to have anything in hand. There is a butter knife on the windowsill.
3. We seem to be holding the butter knife now, beacuse it is at the bottom of the screen, almost as if it was lying on the floor. We are near a microwave.
4. We now see the microwave is turned on. We are probaly heating up the butter knife, since we do not have it in our hand anymore.
5. We have pulled out the butter knife from the microwave. We are now near the windowsill again.

# TASK"""[1:]


def write_config(tasks, output_dir):
    # Config
    with open(output_dir / "configs" / "alfred.yaml", "w") as f:
        config = {
            "tasks": tasks,
            "models": [
                {"kind": "encoder", "encoder": "s3d", "heads": [{"kind": "cosine"}]},
                {"kind": "gpt", "n_frames": 5},
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
