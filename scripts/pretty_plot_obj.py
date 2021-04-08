"""
Plots train-val curves logged in tensorboard (e.g. extracts multiple keys)
"""
#%%
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
tf_size_guidance = {'scalars': 500}

run_root = "/nethome/jye72/projects/embodied-recall/tb/objectnav/"

def get_run_logs_simple(v):
    return [os.path.join(run_root, v)]

# * Key
plot_key = 'eval_success' # spl, success, seval_spl, eval_success
# plot_key = 'eval_coverage' # spl, success, eval_spl, eval_success
# plot_key = 'eval_spl' # spl, success, eval_spl, eval_success
plot_keys = ["success"]
plot_keys = ["success", "eval_success"] # , "spl", "eval_spl"]


variants = [
    "base-full/gt_sem",
    "base-full/pred_sem",
    # "base4-full/gt_sem",
    # "base4-full/pred_sem",
]

colors = defaultdict(lambda x: 'cornflowerblue')
colors_index = 0
palette = sns.color_palette(palette='muted', n_colors=5, desat=0.9)

def get_plot_steps_and_values(key, variants=variants): # Wrapper
    used_variants = variants
    if "eval" not in key:
        used_variants = [v[:v.rfind("/")] for v in variants]
    plot_key_folder = plot_key_folder_dict.get(key, "")
    def get_variant_paths_and_colors(input_variants, key):
        # Set what to plot
        plotted_union = list(input_variants)
        paths = {}
        for i, v in enumerate(plotted_union):
            # Check if stem in colors
            if "eval" not in key:
                stem = v
            else:
                stem = v[:v.rfind("/")]
            if stem in colors:
                colors[v] = colors[stem]
            else:
                global colors_index
                colors[v] = palette[colors_index]
                colors[stem] = colors[v]
                colors_index += 1
        for v in input_variants:
            paths[v] = get_run_logs_simple(v)

        return paths

    variant_paths = get_variant_paths_and_colors(used_variants, key)

    # Load
    def get_tb_metrics(all_paths, key, predicted=False):
        all_values = defaultdict(list)
        all_steps = defaultdict(list)
        # ! TODO guard for duplicate variant stems
        for variant, variant_runs in all_paths.items():
            print(variant, variant_runs)
            # if 'eval' not in key: # Trim leaf
            #     variant = variant[:variant.rfind("/")]
            #     variant_runs = [p[:p.rfind("/")] for p in variant_runs]
            min_steps = 0
            for i, run in enumerate(variant_runs):
                if len(all_steps[variant]) >= run_count:
                    break
                accum_path = run
                # if predicted:
                #     acscum_path = osp.join(accum_path, "pred_sem")
                if 'eval' in key:
                    accum_path = osp.join(accum_path, plot_key_folder)
                else:
                    accum_path = osp.join(accum_path, f"metrics_{key}")
                if not osp.exists(accum_path):
                    print(f"{accum_path} DNE")
                    continue
                event_acc = EventAccumulator(accum_path, tf_size_guidance)
                event_acc.Reload()

                if 'eval' in key:
                    scalars = event_acc.Scalars('eval_metrics')
                else:
                    scalars = event_acc.Scalars("metrics")
                steps_and_values = np.stack(
                    [np.asarray([scalar.step, scalar.value])
                    for scalar in scalars])
                steps = steps_and_values[:, 0]
                values = steps_and_values[:, 1]
                all_steps[variant].append(steps)
                all_values[variant].append(values)
        return all_steps, all_values

    plot_steps, plot_values = get_tb_metrics(variant_paths, key)
    return plot_steps, plot_values, used_variants

dict_steps = {}
dict_values = {}
dict_variants = {}
for key in plot_keys:
    dict_steps[key], dict_values[key], dict_variants[key] = get_plot_steps_and_values(key)

#%%
# * Cropping (and averaging)
#  values of each checkpoint
def get_cleaned_data(raw_steps, raw_values, raw_variants, average=1, warn=False):
    clean_steps = defaultdict(list)
    clean_values = defaultdict(list)
    for variant in raw_variants:
        print(variant) # It's been trimmed!
        if variant in raw_steps:
            for i in range(len(raw_steps[variant])):
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
# if 'eval' in plot_key:
clean_steps_dict = {}
clean_values_dict = {}
for key in plot_keys:
    clean_steps_dict[key], clean_values_dict[key] = get_cleaned_data(dict_steps[key], dict_values[key], dict_variants[key], average=3)
# clean_steps, clean_values = get_cleaned_data(dict_steps[plot_keys[0]], dict_values[plot_keys, average=3)


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

plot_means_dict = {}
plot_ci_dict = {}
true_means_dict = {}
true_ci_dict = {}
for key in plot_keys:
    data = clean_values_dict[key]
    plot_means_dict[key], plot_ci_dict[key] = get_means_and_ci(data, window_size=1, early_stop=True)
    true_means_dict[key], true_ci_dict[key] = get_means_and_ci(data, window_size=1, early_stop=False) # For AUC calc


#%%
# Style
SMALL_SIZE = 10
MEDIUM_SIZE = 12
LARGE_SIZE = 15
plt.style.use('seaborn-muted')
plt.figure(figsize=(6,4))

prep_plt()
plt.ylabel("Success")
# plt.ylabel(key_labels[plot_keys[0]].split('-')[0])
ax = plt.gca()
ax.spines['right'].set_alpha(0)
ax.spines['top'].set_alpha(0)

# Plot evals
# Axes
plt.xlim(40, 130)
# plt.xticks(np.arange(40, 115, 20))
x_scale = 1e6

# if 'eval_s' in plot_keys[0]:
lower_lim = 0.0
upper_lim = 0.8
# upper_lim = 1.0

plt.ylim(lower_lim, upper_lim)
plt.yticks(np.arange(lower_lim, upper_lim + 0.01, 0.1))

plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE, color=(0.1, 0.1, 0.1, .85))    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE, color=(.1, .1, .1, .85))    # fontsize of the tick labels
plt.gca().xaxis.label.set_color((0.1, 0.1, 0.1, .85))
plt.gca().yaxis.label.set_color((0.1, 0.1, 0.1, .85))

plt.xlabel("Steps (Million)")

plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
spine_alpha = 0.2
leg_start = 0.78

local_labels = {
    "etn_cp_long-explore": "Base",
    "etn_cp_im4-explore": "Comm @ 4",
    "split_120-curric": "Split"
}

local_labels = {
    "split-curric": "Split",
    "split-curric/pred_sem": "Split",
    "split-curric/gt_sem": "Split GT",
    "split_120-curric": "Split",
    "split_120-curric/pred_sem": "Split",
    "split_120-curric/gt_sem": "Split GT",
}

def get_label(variant, key):
    if "eval" in key:
        if "pred_sem" in variant:
            return "Val (RedNet Segm)"
        else:
            return "Val (GT Segm)"
    return "Train"

plotted = set()
def plot_all(key):
    # print(variants)

    for variant in dict_variants[key]:
        if (variant, key) in plotted:
            continue
        plotted.add((variant, key))
        # if 'eval' in plot_key:
        # print(clean_steps)
        x = clean_steps_dict[key][variant][0] / x_scale
        # else:
            # x = desired_steps
        # if '/' in key:
        #     y = plot_means_dict[key][variant]
        #     print(y)
        # else:
        y = true_means_dict[key][variant]
        # print(key, variant, y)
        # print(f"{variant} {y[-1]:.3f} @ {clean_steps[variant][0][np.argmax(true_means[variant])]}")
        style = "-" if 'gt' not in variant else "--"
        if "eval" not in key:
            style = ':'
        # style = "-" if 'gt' not in variant else "--"

        # if len(variant.split("/")) > 1 and variant.split("/")[1] == "pred_sem":
            # style = "-"
        # line, = plt.plot(x, y, label=f" {'Val - GT Sem' if 'eval' in key else 'Train'}", c=dict_colors[key].get(variant), linestyle=style)
        line, = plt.plot(x, y, label=get_label(variant, key),
            c=colors.get(variant),
            linestyle=style
        )
        # line, = plt.plot(x, y, label=get_variant_labels(variant, local_labels) + f" {'Val' if 'eval' in key else 'Train'}", c=colors.get(variant), linestyle=style)
        # plt.scatter(x, y, c=variant_colors.get(variant), linestyle=style)
        plt.fill_between(x, y - plot_ci_dict[key][variant], y + plot_ci_dict[key][variant], facecolor=line.get_color(), alpha=0.5)

for key in plot_keys:
    plot_all(key)
# plot_all('success')
# plot_all('eval_success')
# plt.title("Best Agent")

# plt.annotate("", xy=(110, 0.51), xycoords="data", xytext=(110, 0.35), textcoords="data", arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0", linewidth="1.8", color=(0.3, 0.3, 0.3)), size=14)
# plt.text(idx+hoffset, hi+voffset, value_str, size=16)
# plt.text(105, 0.55, "-0.16", size=18)
# plt.vlines(60, 0, 1.0, # , label="Policy Switch", # linestyle='-.',
# color="#333333", alpha=0.5)
# plt.text(55, 0.6, "Policy Switch", rotation=90)

leg = plt.legend(loc=(.6, 0.04),
    markerfirst=False, ncol=1, frameon=False, labelspacing=0.4)
for line in leg.get_lines():
    line.set_linewidth(2.0)

plt.savefig('test.pdf', dpi=150, bbox_inches="tight")
