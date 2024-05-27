# %%
# %load_ext autoreload
# %autoreload 2

# %%
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import polars as pl
from alfred import object, utils
from alfred.task import (
    Task,
    actions_to_blocks,
    clean_tasks,
    container_tasks,
    cool_tasks,
    heat_tasks,
    object_tasks,
    on_v_off_tasks,
    permutation_task,
    pick_up_tasks,
    remixed_task,
    slice_tasks,
    sliced_v_whole_tasks,
    toggle_tasks,
    write_config,
)
from alfred.trajectory import Trajectory, shorten_trajectories
from plotnine import *
from polars import col as c
from rich.pretty import pprint

utils.set_seed(42)

# %%
trajectories = utils.load_trajectories(Path("metadata"))
groups_per_level = {
    level: sorted(
        utils.group_by(
            shorten_trajectories(trajectories, level), lambda t: tuple(t.actions)
        ),
        key=lambda g: len(g),
        reverse=True,
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
benchmark_path = Path("/Users/eugen/Downloads/Projects/mats/vlm-benchmark")
easy_tasks = [t for t in trajectories if t.type != "pick_and_place_with_movable_recep"]

# %%
tasks = []


def limit(tasks, count=5):
    random.shuffle(tasks)
    return tasks[:count]


for t in limit(pick_up_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(clean_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in object_tasks(easy_tasks):
    tasks.append(t.write(benchmark_path))

for t in container_tasks(easy_tasks):
    tasks.append(t.write(benchmark_path))

for t in limit(heat_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(cool_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(toggle_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(slice_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(sliced_v_whole_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

for t in limit(on_v_off_tasks(trajectories)):
    tasks.append(t.write(benchmark_path))

# for level in [3, 6]:
#     random.shuffle(groups_per_level[level])
#     for g in groups_per_level[level][:3]:
#         perm = Task.create_permuted(g, 10)
#         tasks.append(perm.write(Path(benchmark_path)))

#         sub = Task.create_substituted(g, action_library, 10)
#         tasks.append(sub.write(Path(benchmark_path)))

# %%
import math

# tasks = []

tasks_per_level = 33
num_classes = 3
for level in range(2, 9):
    level_num_classes = min(num_classes, math.factorial(level))
    groups = groups_per_level[level]
    random.shuffle(groups)

    consistent_candidates, inconsistent_candidates = [], []
    for g in groups:
        try:
            task = permutation_task(g[0], level_num_classes, trajectories)
        except:
            continue

        if len(task.trajectories) == level_num_classes:
            assert task.metadata is not None
            if task.metadata["is_consistent_wrt_slicing"]:
                consistent_candidates.append(task)
            else:
                inconsistent_candidates.append(task)
        if len(consistent_candidates) >= tasks_per_level:
            break
    else:
        print(
            f"Not enough consistent tasks for level {level}: {len(consistent_candidates)}"
        )
        to_add = tasks_per_level - len(consistent_candidates)
        print(f"Adding {to_add} inconsistent tasks")
        if to_add > len(inconsistent_candidates):
            print(
                f"Only writing {len(consistent_candidates) + len(inconsistent_candidates)} out of {tasks_per_level} tasks"
            )

    for task in (consistent_candidates + inconsistent_candidates)[:tasks_per_level]:
        name = task.write(benchmark_path)
        tasks.append(name)
        print(name)

videos_per_task = 8

level_to_num_videos = {
    2: (4, 4),
    3: (2, 3),
    4: (2, 2),
    5: (4, 1),
    6: (3, 1),
    7: (2, 1),
    8: (1, 1),
}

assert all(
    ((a + (level - 1) * b) == videos_per_task)
    for level, (a, b) in level_to_num_videos.items()
)

tasks_per_level = 12

for level in range(2, 9):
    groups = groups_per_level[level]
    random.shuffle(groups)

    candidates = {}
    for g in groups:
        # Half of the tasks will not have slicing mixed in
        # because we want to see other actions used, too
        if len(candidates.get(videos_per_task + 1, [])) % 2 == 0:
            all_trajectories = [
                t
                for t in trajectories
                if not any(a.action == "SliceObject" for a in t.actions)
            ]
        else:
            all_trajectories = trajectories

        videos_for_prefix_0, videos_per_prefix = level_to_num_videos[level]
        task = remixed_task(
            g[0], videos_for_prefix_0, videos_per_prefix, all_trajectories
        )

        # +1 because we have 8 remixed classes on top of the 1 original class
        candidates.setdefault(len(task.trajectories), []).append(task)
        if len(candidates.get(videos_per_task + 1, [])) >= tasks_per_level:
            break
    else:
        print(
            f"Not enough long tasks for level {level}: {len(candidates.get(videos_per_task + 1, []))}"
        )
        lens = {
            i: len(candidates.get(i, []))
            for i in list(range(videos_per_task + 1, 1, -1))
        }
        print(f"Adding shorter tasks: {lens}")

    final_candidates = sum(
        [candidates.get(i, []) for i in list(range(videos_per_task + 1, 1, -1))], []
    )

    for task in final_candidates[:tasks_per_level]:
        name = task.write(benchmark_path)
        tasks.append(name)
        print(name)

# %%
write_config(tasks, benchmark_path)
