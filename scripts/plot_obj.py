#%%
# This notebook plots tensorboard logs and extracts table numbers.
# Cell 1 does raw extraction. Only extracts one key at a time
# Cell 2 does some processing for redundant logs
# Cell 3+ plots
import math
from collections import defaultdict
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from obj_consts import (
    get_variant_labels,
    key_labels,
    axis_labels,
    plot_key_folder_dict
)

from analyze_utils import prep_plt
run_count = 1
tf_size_guidance = {'scalars': 1000}

run_root = "/nethome/jye72/projects/embodied-recall/tb/objectnav/"

def get_run_logs(v):
    folder = os.path.join(run_root, v)
    run_folders = os.listdir(folder)
    run_folders.sort()
    event_paths = []
    for run_folder in run_folders:
        full_path = os.path.join(folder, run_folder)
        if os.path.isdir(full_path) and 'run' in run_folder:
            event_paths.append(full_path)
    return event_paths

def get_run_logs_simple(v):
    return [os.path.join(run_root, v)]

# * Key
plot_key = 'eval_success' # spl, success, eval_spl, eval_success
# plot_key = 'eval_distance_to_goal' # spl, success, eval_spl, eval_success
plot_key = 'eval_spl' # spl, success, eval_spl, eval_success
# plot_key = 'eval_coverage' # spl, success, eval_spl, eval_success
# plot_key = 'eval_steps'

plot_key_folder = plot_key_folder_dict.get(plot_key, "")

variants = [
    # "base-full/gt_sem", # done
    # "base-full/pred_sem", # done
    # "base4-full/gt_sem", # done
    # "base4-full/pred_sem", # done
    # "split_clamp-full/gt_sem",
    # "split_clamp-full/pred_sem",

    # "no_cp-full/pred_sem",
    "no_adp-full/pred_sem",
    "no_gid-full/pred_sem",
    # "no_sge-full/pred_sem",
    "aux_sge-full/pred_sem",
    # "pt_sparse-full/gt_sem",
    # "pt_sparse-full/pred_sem",
]

def get_variant_paths_and_colors(input_variants):
    # Set what to plot
    plotted_union = list(input_variants)
    stems = list(set(v.split("/")[0] for v in plotted_union))
    palette = sns.color_palette(palette='muted', n_colors=len(stems), desat=0.9)

    colors = {}
    for v in plotted_union:
        # Get the stem index
        stem = v.split("/")[0]
        colors[v] = palette[stems.index(stem)]

    sns.palplot(palette)
    paths = {}
    for v in input_variants:
        paths[v] = get_run_logs_simple(v)

    return paths, colors

variant_paths, variant_colors = get_variant_paths_and_colors(variants)

# Load
def get_tb_metrics(all_paths, predicted=False):

    all_values = defaultdict(list)
    all_steps = defaultdict(list)
    for variant, variant_runs in all_paths.items():
        min_steps = 0
        for i, run in enumerate(variant_runs):
            if len(all_steps[variant]) >= run_count:
                break
            accum_path = run
            # if predicted:
            #     acscum_path = osp.join(accum_path, "pred_sem")
            if 'eval' in plot_key:
                accum_path = os.path.join(accum_path, plot_key_folder)
            elif plot_key == "spl":
                accum_path = os.path.join(accum_path, "metrics_spl")
            if not os.path.exists(accum_path):
                continue
            event_acc = EventAccumulator(accum_path, tf_size_guidance)
            event_acc.Reload()

            if 'eval' in plot_key:
                scalars = event_acc.Scalars('eval_metrics')
            elif plot_key == "success":
                scalars = event_acc.Scalars(plot_key)
            elif plot_key == "spl":
                scalars = event_acc.Scalars("metrics")
            steps_and_values = np.stack(
                [np.asarray([scalar.step, scalar.value])
                for scalar in scalars])
            steps = steps_and_values[:, 0]
            values = steps_and_values[:, 1]
            all_steps[variant].append(steps)
            all_values[variant].append(values)
    return all_steps, all_values

plot_steps, plot_values = get_tb_metrics(variant_paths)

#%%
# * Cropping (and averaging) values of each checkpoint.
# Averaging only used if there are multiple validation runs (not used in this instance.)
def get_cleaned_data(raw_steps, raw_values, average=1, warn=False):
    clean_steps = {}
    clean_values = {}
    for variant in variants:
        clean_steps[variant] = []
        clean_values[variant] = []
        if variant in plot_steps:
            for i in range(len(plot_steps[variant])):
                steps = raw_steps[variant][i]
                vals = raw_values[variant][i]
                un, ind, inv = np.unique(steps, return_index=True, return_inverse=True)
                # all the places where there are 0s, is where the first unique is. Select them
                clean_steps[variant].append(steps[ind])
                avg_values = []
                for step in range(len(un)):
                    if warn and len(vals[inv == step]) < average:
                        print(f"Only {len(vals[inv == step])} runs for {variant}, step {step}")
                    step_vals = vals[inv == step][:average]
                    avg_step_val = np.mean(step_vals)
                    avg_values.append(avg_step_val)
                print(variant, i, len(avg_values))
                clean_values[variant].append(avg_values)
        print(np.array(clean_values[variant]).shape)
    return clean_steps, clean_values
if 'eval' in plot_key:
    clean_steps, clean_values = get_cleaned_data(plot_steps, plot_values, average=3)

def get_means_and_ci(values, window_size=1, early_stop=True):
    r"""
        Returns means and CI np arrays
        args:
            values: dict of trials by variant, each value a list of trial data
            window_size: window smoothing of trials
        returns:
            mean and CI dict, keyed by same variants
    """
    means={}
    ci = {}
    for variant in values:
        data = np.array(values[variant])
        # print(data.shape)
        # print(data.shape)
        # print(variant)
        values_smoothed = np.empty_like(data)
        if window_size > 1:
            for i in range(data.shape[1]):
                window_start = max(0, i - window_size)
                window = data[:, window_start:i + 1]
                values_smoothed[:, i] = window.mean(axis=1)
        else:
            values_smoothed = data

        if early_stop:
            best_until = np.copy(values_smoothed)
            for t in range(best_until.shape[1]):
                best_until[:,t] = np.max(best_until[:,:t+1], axis=1)
            values_smoothed = best_until

        means[variant] = np.mean(values_smoothed, axis=0)
        ci[variant] = 1.96 * np.std(values_smoothed, axis=0) \
            / math.sqrt(run_count) # 95%
    return means, ci
data = clean_values

plot_means, plot_ci = get_means_and_ci(data, window_size=1, early_stop=True)
plot_means, plot_ci = get_means_and_ci(data, window_size=1, early_stop=False)
true_means, true_ci = get_means_and_ci(data, window_size=1, early_stop=False) # For AUC calc
val_means, val_ci = plot_means, plot_ci

#%%
# Style
prep_plt()
plt.ylabel(key_labels[plot_key])

# Plot evals
# Axes
# plt.xlim(40, 80)
# plt.xlim(40, 110)
plt.xticks(np.arange(20, 130, 20))
x_scale = 1e6

if plot_key in ['eval_spl', 'eval_success']:
    lower_lim = 0.0
    upper_lim = 0.6

    plt.ylim(lower_lim, upper_lim)
    plt.yticks(np.arange(lower_lim, upper_lim + 0.01, 0.1))

# plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE, color=(0.1, 0.1, 0.1, .85))    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE, color=(.1, .1, .1, .85))    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.gca().xaxis.label.set_color((0.1, 0.1, 0.1, .85))
plt.gca().yaxis.label.set_color((0.1, 0.1, 0.1, .85))
sns.despine(ax=plt.gca())
plt.xlabel("Steps (Million)")

spine_alpha = 0.2
leg_start = 0.78
plotted = variants

local_labels = {
    "etn_cp_long-explore": "Coverage then D2G",
    "etn_cp_24gb-explore": "2x Batch",
    "no_rgb-curpol": "No RGB",
    "no_sge-sge": "No Goal Detection",
    "merged_vision4-sem2": "Coverage + D2G"
}

for variant in plotted:
    if 'eval' in plot_key:
        x = clean_steps[variant][0] / x_scale
    else:
        x = desired_steps
    y = plot_means[variant]
    print(x)
    print(y)
    argmax = np.argmax(y)
    print(f"{variant} {y[argmax]:.3f} @ {x[argmax]:.3g}")
    style = "--"
    if len(variant.split("/")) > 1 and variant.split("/")[1] == "pred_sem":
        style = "-"
    line, = plt.plot(x, y, label=get_variant_labels(variant, local_labels), c=variant_colors.get(variant), linestyle=style)
    plt.scatter(x, y, c=variant_colors.get(variant), linestyle=style)
    plt.fill_between(x, y - plot_ci[variant], y + plot_ci[variant], facecolor=line.get_color(), alpha=0.5)

def annotate(idx, from_var, to_var, hoffset=-6, voffset=0, extender=0.02): # extender for graphical tweaks
    lo = plot_means[from_var][idx-1] - extender
    hi = plot_means[to_var][idx-1]
    # plt.annotate("", xy=(idx, lo), xycoords="data", xytext=(idx, hi), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", linewidth="1.8"))
    plt.annotate("", xy=(idx, lo), xycoords="data", xytext=(idx, hi), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", linewidth="1.8", color=(0.3, 0.3, 0.3)), size=14)
    value = hi - lo - extender
    value_str = f"{value:.2f}"
    if value > 0:
        value_str = "+" + value_str
    # plt.text(idx+hoffset, hi+voffset, value_str, size=16)
    plt.text(idx+hoffset, hi+voffset, value_str + " SPL", size=14)

leg_start = 1.0
if plot_key == 'eval_spl':
    height = 0.5
else:
    height = 1.0
leg = plt.legend(loc=(.04, height),
    markerfirst=True, ncol=1, frameon=False, labelspacing=0.4)
for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.savefig('test.pdf', dpi=150, bbox_inches="tight")


# SPL
# base-full gt: 0.157 @ 112 (31)
# base-full pred: 0.076 @ 112 (31)
# base4-full gt: 0.141 @ 123 (34)
# base4-full pred: 0.096 @ 119 (33)
# split_clamp gt: 0.215 @ 113 (31)
# split_clamp pred: 0.094 @ 113 (31)

# pt_sparse pred: 0.084 @ 123 (34)
# no_adp pred: 0.093 @ 116 (32)
# np_cp pred: 0.088 @ 124 (34)

# aux_sge pred: 0.04 @ 131 (36)

# Succ (indexed by best SPL)
# base-full gt: 0.554
# base-full pred: 0.308
# base4-full gt: 0.474
# base4-full pred: 0.344
# split_clamp gt: 0.522 @ 113
# split_clamp pred: 0.260 @ 113

# pt_sparse pred: 0.231
# no_adp pred: 0.341

#%%
# Extract episode level statistics to calculate CI.
from analyze_utils import load_stats
from fp_finder import load_variant
import scipy.stats as st
is_gt = False
variant = 'base4-full'
variant = 'base4_rednet-full' # 36
variant = 'base-full' # 31
variant = 'base_rednet-full' # 36
variant = 'split_clamp-full' # 31
variant = 'split_rednet-full' # 37

variant = 'no_adp-full' # 32
variant = 'no_gid-full' # 35
variant = 'no_cp-full' # 34
variant = 'pt_sparse-full' # 34
variant = 'no_sge-full' # 35

variant = 'aux_sge-full' # 36

ckpt = 36
is_gt = False
# meta_df, title = load_stats(variant, ckpt, is_gt=is_gt)
# meta_df, title = load_stats(variant, ckpt, override_fn=f'{variant}/{ckpt}/eval_gt_False_21.pth')

# GT Segmentation table
# meta_df, title = load_stats(variant, ckpt, override_fn=f'{variant}/{ckpt}/eval_gt_False_21.pth')
variant = 'base4-full' # 33, 34 (35) - for GT true since 34 is anomalously bad
variant = 'base-full' # 31
variant = 'split_clamp-full' # 31
ckpt = 31
meta_df = load_variant(variant, ckpt, override_fn=f'{variant}/{ckpt}/train_gt_False_21.pth')
# meta_df = load_variant(variant, ckpt, override_fn=f'{variant}/{ckpt}/train_gt_True.pth')
# meta_df = load_variant(variant, ckpt, override_fn=f'{variant}/{ckpt}/eval_gt_True.pth')
# meta_df = load_variant(variant, ckpt, override_fn=f'{variant}/{ckpt}/eval_gt_False.pth')

def print_info(df):
    succ = df['success']
    spl = df['spl']
    def get_ci(data): # in percent
        return (np.mean(data) - st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]) * 100
    print(f"{variant} Success")
    print(f"${(succ.mean()*100):.3g} $\scriptsize{{$\pm {get_ci(succ):.2g}$}}")
    print(f"{variant} SPL")
    print(f"${(spl.mean()*100):.3g} $\scriptsize{{$\pm {get_ci(spl):.2g}$}}")

print_info(meta_df)