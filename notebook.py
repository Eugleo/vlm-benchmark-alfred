# %%
# %load_ext autoreload
# %autoreload 2

# %%
import math
import random
from pathlib import Path

from alfred import utils
from alfred.task import (
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
from alfred.trajectory import shorten_trajectories
from plotnine import *

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
benchmark_path = Path("test")
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


# %%
permutation_tasks_per_level = 50
num_classes = 3
for level in range(4, 9):
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
        if len(consistent_candidates) >= permutation_tasks_per_level:
            break
    else:
        print(
            f"Not enough consistent tasks for level {level}: {len(consistent_candidates)}"
        )
        to_add = permutation_tasks_per_level - len(consistent_candidates)
        print(f"Adding {to_add} inconsistent tasks")
        if to_add > len(inconsistent_candidates):
            print(
                f"Only writing {len(consistent_candidates) + len(inconsistent_candidates)} out of {permutation_tasks_per_level} tasks"
            )

    for task in (consistent_candidates + inconsistent_candidates)[
        :permutation_tasks_per_level
    ]:
        name = task.write(benchmark_path)
        tasks.append(name)
        print(name)

# %%
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


remix_tasks_per_level = 19

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
        if len(candidates.get(videos_per_task + 1, [])) >= remix_tasks_per_level:
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

    for task in final_candidates[:remix_tasks_per_level]:
        name = task.write(benchmark_path)
        tasks.append(name)
        print(name)

# %%
write_config(tasks, benchmark_path)
