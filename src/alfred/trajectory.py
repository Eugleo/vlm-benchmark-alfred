import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from alfred import object


@dataclass
class LowLevelAction:
    images: list[int]
    action: str
    object1: Optional[str] = None
    object2: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.action, self.object1, self.object2))

    def __eq__(self, other) -> bool:
        return (
            self.action == other.action
            and self.object1 == other.object1
            and self.object2 == other.object2
        )

    @staticmethod
    def from_json(js: dict, images: list[int]) -> "LowLevelAction":
        action = js["api_action"]["action"]
        if action in [
            "PickupObject",
            "OpenObject",
            "PutObject",
            "CloseObject",
            "ToggleObjectOn",
            "ToggleObjectOff",
        ]:
            object1 = js["api_action"]["objectId"].split("|")[0]
        else:
            object1 = None

        if action in ["PutObject"]:
            object2 = js["api_action"]["receptacleObjectId"].split("|")[0]
        else:
            object2 = None

        return LowLevelAction(
            action=action, images=images, object1=object1, object2=object2
        )


@dataclass
class HighLevelAction:
    human_descriptions: list[str]
    low_level_actions: list[LowLevelAction]

    action: str
    object1: str
    object2: Optional[str] = None

    # Only used for substitutions
    # If None, we inherit the image path from the video we're in
    _images_path: Optional[Path] = None

    def __hash__(self) -> int:
        return hash((self.action, self.object1, self.object2))

    def __eq__(self, other) -> bool:
        return (
            self.action == other.action
            and self.object1 == other.object1
            and self.object2 == other.object2
        )

    @property
    def description(self):
        if self.action == "PickupObject":
            return f"we pick up a {self.object1}"
        elif self.action == "ToggleObject":
            return f"we turn the {self.object1} on"
        elif self.action == "SliceObject":
            return f"we slice the {self.object1}"
        elif self.action == "CleanObject":
            return f"we clean the {self.object1}"
        elif self.action == "GotoLocation":
            return f"we walk to the {self.object1}"
        else:
            assert self.object2 is not None
            preposition = object.preposition.get(self.object2, "on")
            if self.action == "PutObject":
                return f"we put the {self.object1} {preposition} the {self.object2}"
            elif self.action == "CoolObject":
                return f"we cool the {self.object1} {preposition} the {self.object2}"
            elif self.action == "HeatObject":
                return f"we heat the {self.object1} {preposition} the {self.object2}"
            else:
                raise ValueError(f"Unknown action: {self.action}")

    @staticmethod
    def from_json(
        js: dict, descriptions: list[str], low_level_actions: list[LowLevelAction]
    ) -> "HighLevelAction":
        action = js["planner_action"]["action"]
        if action in ["HeatObject", "CoolObject", "GotoLocation"]:
            object1 = js["discrete_action"]["args"][0]
        else:
            object1 = js["planner_action"]["coordinateObjectId"][0]
        if object1 not in object.id_to_str:
            raise ValueError(f"Unknown object id: {object1}")
        object1 = object.id_to_str[object1]

        if action in ["PutObject", "HeatObject", "CoolObject"]:
            object2 = js["planner_action"]["coordinateReceptacleObjectId"][0]
            if object2 not in object.id_to_str:
                raise ValueError(f"Unknown object id: {object2}")
            object2 = object.id_to_str[object2]
        else:
            object2 = None

        return HighLevelAction(
            human_descriptions=descriptions,
            low_level_actions=low_level_actions,
            action=action,
            object1=object1,
            object2=object2,
        )


@dataclass
class Trajectory:
    trial_id: str
    dataset: str
    type: str

    actions: list[HighLevelAction]

    _human_overall_annotations: list[str]
    _images_path: Path

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.with_modified_actions(self.actions[idx])
        else:
            return self.actions[idx]

    @property
    def description(self, high_level=True) -> str:
        if high_level:
            actions = self.actions
            last_action = actions[-1].description
            return (
                "First, "
                + ", then ".join(a.description for a in actions[:-1])
                + (", and finally " if len(actions) > 2 else ", and then ")
                + last_action
            )
        else:
            raise NotImplementedError

    @property
    def video(self) -> ImageSequenceClip:
        clip_cache, frames = {}, []
        for hla in self.actions:
            path = hla._images_path or self._images_path
            clip = clip_cache.setdefault(path, VideoFileClip(str(path)))
            frame_indices = [img for lla in hla.low_level_actions for img in lla.images]
            fps = clip.fps
            frames += [clip.get_frame(i / fps) for i in frame_indices]
        return ImageSequenceClip(frames, fps=fps)

    def substitute(self, index: int, action: HighLevelAction) -> "Trajectory":
        if index > len(self):
            raise ValueError("Index out of range")
        return self.with_modified_actions(
            self.actions[:index] + [action] + self.actions[index + 1 :]
        )

    def filter_low_level(self, predicate) -> "Trajectory":
        new_actions = []
        for action in self.actions:
            new_low_level_actions = [
                lla for lla in action.low_level_actions if predicate(lla)
            ]
            new_actions.append(
                HighLevelAction(
                    action.human_descriptions,
                    new_low_level_actions,
                    action.action,
                    action.object1,
                    action.object2,
                )
            )
        return self.with_modified_actions(new_actions)

    def with_modified_actions(self, actions: list[HighLevelAction]):
        return Trajectory(
            self.trial_id, self.dataset, self.type, actions, [], self._images_path
        )

    @staticmethod
    def from_file(path: Path):
        with open(path, "r") as f:
            js = json.load(f)

            high_to_low, low_to_images = {}, {}
            for i, img in enumerate(js["images"]):
                high_to_low.setdefault(img["high_idx"], set()).add(img["low_idx"])
                low_to_images.setdefault(img["low_idx"], []).append(i)

            action_count = len(js["turk_annotations"]["anns"][0]["high_descs"])
            descriptions = [
                [desc["high_descs"][i] for desc in js["turk_annotations"]["anns"]]
                for i in range(action_count)
            ]

            low_level_actions = [
                LowLevelAction.from_json(action, low_to_images[i])
                for i, action in enumerate(js["plan"]["low_actions"])
            ]
            high_level_actions = [
                HighLevelAction.from_json(
                    action,
                    description,
                    [
                        low_level_actions[i]
                        for i in sorted(list(high_to_low[action["high_idx"]]))
                    ],
                )
                for action, description in zip(js["plan"]["high_pddl"], descriptions)
            ]

            # Here we assume that:
            # - the video dir is videos
            # - other than that, the video is stored in a path that mirrors `path`
            # - the video is named video.mp4
            parts = list(path.parts)
            parts[0] = "videos"
            images_path = list(Path(*parts[:-1]).glob("**/video.mp4"))[0]

            return Trajectory(
                trial_id=js["task_id"],
                dataset=parts[1],
                type=js["task_type"],
                actions=high_level_actions,
                _human_overall_annotations=[
                    a["task_desc"] for a in js["turk_annotations"]["anns"]
                ],
                _images_path=images_path,
            )


def shorten_trajectories(trajectories: list[Trajectory], to: int) -> list["Trajectory"]:
    return [
        t
        for trajectory in trajectories
        for t in [trajectory[beg : beg + to] for beg in range(len(trajectory) - to + 1)]
    ]
