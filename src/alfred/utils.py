from pathlib import Path

from alfred.trajectory import Trajectory


def group_by(objects, fun):
    result = {}
    for obj in objects:
        result.setdefault(fun(obj), []).append(obj)
    return list(result.values())


def get_action_tuples(trajectory: Trajectory):
    return [(a.action, a.object1, a.object2) for a in trajectory.actions]


def set_action_tuples(trajectory: Trajectory, tuples):
    for a, t in zip(trajectory.actions, tuples):
        a.action, a.object1, a.object2 = t


def load_trajectories(metadata_path: Path) -> list[Trajectory]:
    trajectories = []
    for trajectory_path in metadata_path.glob("**/traj_data.json"):
        try:
            trajectory = Trajectory.from_file(trajectory_path)
        except Exception as e:
            print("Error loading", trajectory_path, e)
            continue
        trajectories.append(trajectory)
    return trajectories
