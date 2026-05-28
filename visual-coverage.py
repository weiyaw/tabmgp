# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
import os
import re
import jax

import utils
from experiment_setup import load_experiment, get_experiment_paths
from credible_set import joint_credible_set, coverage_probability

jax.config.update("jax_enable_x64", True)

# %%
# Coverage of TabMGP at various rollout length (N=250,500,750,1000)
LOSS = "likelihood"
ALPHA = 0.05
OUTPUT_DIR = Path("../outputs")
SAVE_DIR = Path("../paper/images")
TABLE_DIR = Path("./table")
SAVE_PLOTS = True

DATE_TITLE = {
	"2025-09-01": "Classification GMM $a=0$",
	# "2025-09-02": "Classification GMM $a=-1$",
	# "2025-09-03": "Classification GMM $a=-2$",
	"2025-09-51": "skin",
	"2025-09-52": "yeast",
	"2025-09-53": "wine",
}


def read_tabpfn_posteriors(seed_path, loss):
	"""
	Return {rollout_length: posterior_samples} for one seed directory.
	"""
	posterior = {}
	for root, _, files in os.walk(seed_path):
		if not root.endswith(f"posterior-{loss}"):
			continue
		for filename in sorted(files):
			if match := re.search(r"^tabpfn-(\d+)-post.pickle", filename):
				rollout_len = int(match.group(1))
				posterior[rollout_len] = utils.read_from(f"{root}/{filename}")[0]
	return posterior


def compute_joint_stats(posteriors, alpha, true_value, cov_type="diag"):
	crs = [joint_credible_set(samples, alpha, cov_type=cov_type) for samples in posteriors]
	rate, _ = coverage_probability(crs, true_value)
	post_cov_trace = np.asarray([cr["trace"] for cr in crs], dtype=float)
	return float(rate), float(np.median(post_cov_trace))


date_dirs = [date for date in DATE_TITLE if (OUTPUT_DIR / date).exists()]

rows = []
for date in date_dirs:
	all_paths = get_experiment_paths(f"{OUTPUT_DIR}/{date}")
	if not all_paths:
		continue

	if not Path(all_paths[0], "dgp.pickle").exists():
		print(f"Skipping {date}: missing dgp.pickle")
		continue

	data_name = utils.get_data_name(all_paths[0])
	_, _, theta_true, _ = load_experiment(all_paths[0], loss=LOSS)
	all_posteriors = [read_tabpfn_posteriors(path, LOSS) for path in all_paths]

	non_empty = [set(p.keys()) for p in all_posteriors if p]
	if not non_empty:
		continue

	rollout_lengths = sorted(set.intersection(*non_empty))

	for rollout_len in rollout_lengths:
		posteriors = [p[rollout_len] for p in all_posteriors if rollout_len in p]
		if not posteriors:
			continue

		coverage, post_cov_trace_median = compute_joint_stats(
			posteriors, ALPHA, theta_true, cov_type="diag"
		)

		rows.append(
			{
				"date": date,
				"data": data_name,
				"N - n": rollout_len,
				"coverage": coverage,
				"size": post_cov_trace_median,
				"ideal_coverage": 1 - ALPHA,
				"n_reps": len(posteriors),
			}
		)

coverage_df = pd.DataFrame(rows)
if coverage_df.empty:
	raise RuntimeError("No TabPFN posterior files found in 2025-09-* outputs.")

coverage_df = coverage_df.sort_values(["date", "N - n"]).reset_index(drop=True)
# TABLE_DIR.mkdir(parents=True, exist_ok=True)
# coverage_df.to_csv(TABLE_DIR / "tabpfn-rollout-coverage.csv", index=False)


# %%
# Plot rollout-length coverage for each 2025-09 setup
sns.set_theme(style="whitegrid")
plot_dates = coverage_df["date"].drop_duplicates().tolist()
n_plots = len(plot_dates)
ncols = n_plots
nrows = 2

fig, axes = plt.subplots(
	nrows,
	ncols,
	figsize=(4.5 * ncols, 6.0),
	squeeze=False,
	sharex="col",
)

for i, date in enumerate(plot_dates):
	date_df = coverage_df[coverage_df["date"] == date].sort_values("N - n")
	x = date_df["N - n"]

	# Top row: coverage
	ax_cov = axes[0, i]
	ax_cov.plot(
		x,
		date_df["coverage"],
		marker="o",
		linewidth=2,
		color=sns.color_palette("colorblind")[0],
	)
	ax_cov.axhline(1 - ALPHA, linestyle="--", color="black", linewidth=1)
	ax_cov.set_title(DATE_TITLE.get(date, date_df["data"].iloc[0]))
	ax_cov.set_ylim(0.8, 1.02)
	ax_cov.set_xticks(sorted(x.unique()))
	if i == 0:
		ax_cov.set_ylabel("Coverage")

	# Bottom row: interval size (post_cov_trace_median)
	ax_size = axes[1, i]
	ax_size.plot(
		x,
		date_df["size"],
		marker="o",
		linewidth=2,
		color=sns.color_palette("colorblind")[1],
	)
	ax_size.set_xlabel("T = N - n")
	ax_size.set_xticks(sorted(x.unique())[::2])
	if i == 0:
		ax_size.set_ylabel("Size")

plt.tight_layout()
if SAVE_PLOTS:
	SAVE_DIR.mkdir(parents=True, exist_ok=True)
	fig.savefig(SAVE_DIR / "sensitivity-rollout-tabmgp.pdf", bbox_inches="tight")
else:
	plt.show()
# plt.close(fig)


# %%




# %%
