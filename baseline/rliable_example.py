import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve

n_seeds = 2
# Read data from different runs
# This is the toy data, you can also build a proper loop over your own runs.
df_s0 = pd.read_csv("demo_data_seed_0.csv")
df_s1 = pd.read_csv("demo_data_seed_1.csv")
# Add a column to distinguish between seeds
# You would do something similar for different algorithms
df_s0["seed"] = 0
df_s1["seed"] = 1
# Combine the dataframes and convert to numpy array

df = pd.concat([df_s0, df_s1], ignore_index=True)
# Make sure only one set of steps is attempted to be plotted
# Obviously the steps should match in such cases!
steps = df["steps"].to_numpy().reshape((n_seeds, -1))[0]
# You can add other algorithms here
train_scores = {"dqn": df["rewards"].to_numpy().reshape((n_seeds, -1))}

# This aggregates only IQM, but other options include mean and median
# Optimality gap exists, but you obviously need optimal scores for that
# If you want to use it, check their code
iqm = lambda scores: np.array(  # noqa: E731
    [metrics.aggregate_iqm(scores[:, eval_idx]) for eval_idx in range(scores.shape[-1])]
)
iqm_scores, iqm_cis = get_interval_estimates(
    train_scores,
    iqm,
    reps=2000,
)

# This is a utility function, but you can also just use a normal line plot with the IQM and CI scores
plot_sample_efficiency_curve(
    steps + 1,
    iqm_scores,
    iqm_cis,
    algorithms=["dqn"],
    xlabel=r"Number of Evaluations",
    ylabel="IQM Normalized Score",
)
plt.gcf().canvas.manager.set_window_title(
    "IQM Normalized Score - Sample Efficiency Curve"
)
plt.legend()
plt.tight_layout()
plt.show()
