# %%
# %load_ext autoreload
# %autoreload 2

# %%
from copy import deepcopy
from pathlib import Path

import polars as pl
from alfred import object, utils
from alfred.task import Task
from alfred.trajectory import shorten_trajectories
from plotnine import *
from polars import col as c

# %%
trajectories = utils.load_normalized_trajectories(Path("metadata"))
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
