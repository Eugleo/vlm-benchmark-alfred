# %%
%load_ext autoreload
%autoreload 2

# %%
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random

import polars as pl
from alfred import object, utils
from alfred.task import Task, object_tasks, container_tasks, write_config,clean_tasks,heat_tasks, cool_tasks, toggle_tasks, pick_up_tasks, slice_tasks,sliced_v_whole_tasks,on_v_off_tasks
from alfred.trajectory import Trajectory, shorten_trajectories
from plotnine import *
from polars import col as c
from rich.pretty import pprint

# %%
trajectories = utils.load_trajectories(Path("metadata"))
groups_per_level = {
    level: utils.group_by(
        shorten_trajectories(trajectories, level), lambda t: tuple(t.actions)
    )
    for level in range(11)
}
# %%
action_library = {}
for action in object.actions:
    for t in trajectories:
        for a in t.actions:
            if a.action == action:
                a = deepcopy(a)
                a._images_path = t._images_path
                action_library.setdefault(action, set()).add(a)


# %%
g = groups_per_level[3][0]
g = g[:2]
t = Task.create_permuted(g, 3)
t.write(Path("test"))

# %%
g = groups_per_level[3][0]
g = g[:2]

t = Task.create_substituted(g, action_library, 3)
t.write(Path("test"))


# %%
types = set(t.type for t in trajectories)
actions = {}
for type in types:
    for t in [t for t in trajectories if t.type == type]:
        actions.setdefault(type, Counter()).update(
            [tuple(a.action for a in t.actions if a.action != "GotoLocation")]
        )
pprint(actions)

# %%
def collapse_moves(actions):
    result = []
    last_was_move = False
    for a in actions:
        if a.startswith("Move") or a.startswith("Rotate") or a.startswith("Look"):
            if last_was_move:
                continue
            last_was_move = True
            a = "Move/Rotate/Look"
        else:
            last_was_move = False
        result.append(a)
    return result

actions = {}
for t in trajectories:
    for a in t.actions:
        actions.setdefault(a.action, Counter()).update(
            [tuple(collapse_moves(lla.action for lla in a.actions))]
        )
pprint(actions)

# %%

# %%
path = Path("metadata/train/pick_and_place_with_movable_recep-Pencil-Mug-Dresser-322")
for p in path.glob("**/traj_data.json"):
    t = Trajectory.from_file(p)
    pprint([a.action for a in t.actions])

# %%
Trajectory.from_file(
    Path(
        "metadata/train/pick_and_place_with_movable_recep-Pencil-Mug-Dresser-322/trial_T20190907_044602_801023/traj_data.json"
    )
)

# %%
benchmark_path = Path("/Users/eugen/Downloads/Projects/mats/vlm-benchmark")
easy_tasks = [t for t in trajectories if t.type != "pick_and_place_with_movable_recep"]

tasks = []

for t in object_tasks(easy_tasks):
    tasks.append(t.write(benchmark_path))

for t in container_tasks(easy_tasks):
    tasks.append(t.write(benchmark_path))

for t in clean_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in heat_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in cool_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in toggle_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in pick_up_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in slice_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in sliced_v_whole_tasks(trajectories):
    tasks.append(t.write(benchmark_path))

for t in on_v_off_tasks(trajectories):
    tasks.append(t.write(benchmark_path))


write_config(tasks, benchmark_path)

# %%
for t in toggle_tasks(trajectories):
    tasks.append(t.write(Path("test")))

# %%
for t in on_v_off_tasks(trajectories):
    tasks.append(t.write(Path("test")))

# %%
tasks = on_v_off_tasks(trajectories)
print(f"Created {len(tasks)} tasks")
pprint(tasks[0].descriptions)
tasks[2].write(Path("test"))
# %%
[
    (hla.action, hla.object1)
    for t in trajectories
    for hla in t.actions
    if "slice" in hla.object1
]
# %%
