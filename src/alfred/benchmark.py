import json

import yaml

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


def write_task(name_prefix, name, activities, descriptions, output_dir):
    tasks_dir = output_dir / "tasks" / "alfred" / name_prefix
    tasks_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos" / "alfred" / name_prefix
    videos_dir.mkdir(parents=True, exist_ok=True)

    description_to_label = {d: f"label_{i}" for i, d in enumerate(descriptions)}
    label_to_description = {v: k for k, v in description_to_label.items()}

    # Task definition
    with open(tasks_dir / f"{name}.yaml", "w") as f:
        yaml.dump({"label_prompts": label_to_description, "prompt_gpt": PROMPT_GPT}, f)

    videos = []
    # Task videos
    for activity in activities:
        activity.video.write_videofile(
            str(videos_dir / f"{activity.trial_id}.mp4"), logger=None
        )

    # Task data
    with open(tasks_dir / f"{name}_data.json", "w") as f:
        videos = [
            {
                "path": f"{name_prefix}/{a.trial_id}.mp4",
                "label": description_to_label[a.description],
            }
            for a in activities
        ]
        json.dump(videos, f)

    return f"{name_prefix}/{name}"


def write_config(tasks, output_dir):
    # Config
    with open(output_dir / "configs" / "alfred.yaml", "w") as f:
        config = {
            "tasks": tasks,
            "models": [
                {
                    "kind": "encoder",
                    "encoder": "s3d",
                    "heads": [{"kind": "cosine"}],
                },
                {
                    "kind": "encoder",
                    "encoder": "clip",
                    "heads": [{"kind": "cosine"}],
                    "hf_model": "ViT-bigG-14/laion2b_s39b_b160k",
                    "n_frames": 8,
                },
                {
                    "kind": "encoder",
                    "encoder": "viclip",
                    "heads": [{"kind": "cosine"}],
                    "batch_size": 16,
                },
            ],
            "task_dir": "tasks/alfred",
            "video_dir": "videos/alfred",
            "cache_dir": ".cache",
            "output_dir": "experiments",
        }
        yaml.dump(config, f)
