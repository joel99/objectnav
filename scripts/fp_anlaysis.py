#%%
# Reproduce fixed point analysis figures
import math
import numpy as np
import numpy.linalg as linalg

import torch
import pandas as pd
import os
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
import scipy.stats as stats

import seaborn as sns
import PIL.Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from analyze_utils import (
    prep_plt, plot_to_axis, get_belief,
    load_device,
    loc_part_ratio, part_ratio, svd
)

from fp_finder import (
    get_variant_path, load_variant, load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path,
    get_jac_pt, get_jac_pt_sequential,
    get_eigvals, time_const
)

from fp_plotter import (
    get_pca_view,
    scatter_pca_points_with_labels,
    add_pca_inset,
    plot_spectrum
)

CUDA_DEVICE = 1 # From alloc-ed item
device = load_device(CUDA_DEVICE)
device = torch.device("cpu")

BELIEF_ABBREVIATIONS = {
    "CPCA": "CPC|A-4",
    "CPCA_B": "CPC|A-16",
    "CoveragePrediction": "CP",
    "ActionDist_A": "ADP",
    "Dummy": "Base",
    "GID": "GID",
}

def get_labels(variant, ckpt):
    ckpt = torch.load(f'/srv/flash1/jye72/share/objectnav/{variant}/{variant}.{ckpt}.pth', map_location='cpu')
    tasks = ckpt['config']['RL']['AUX_TASKS']['tasks']
    return [BELIEF_ABBREVIATIONS.get(t, t) for t in tasks]

def evaluate_memory(model, obs, points, threshold=10):
    # points: B x H
    # TODO add STD
    _, state_grad = get_jac_pt(model, obs, points)
    vals, vecs = get_eigvals(state_grad)
    return (time_const(vals) > threshold).sum(axis=-1).mean()

def memory_analysis_single(variant, ckpt, index, threshold=10, sample_count=50, seed=0, df=None):
    if df is None:
        df = load_variant(variant, ckpt)
    model = load_recurrent_model(variant, ckpt, index=index)
    obs, *_ = get_model_inputs(
        df, scene='all', index=index, size=10000, seed=0
    )
    cache_fps = get_cache_path(variant, ckpt, 10000, index, 'all')
    info = torch.load(cache_fps, map_location='cpu')
    fps = info['fps']
    deltas = info['deltas']
    if deltas.size(0) == 1:
        deltas = deltas[0]
    slow_mask = deltas < 4e-7 # Make sure we're using slower FPs

    np.random.seed(seed)
    subset = fps[slow_mask]
    selected_index = np.random.choice(subset.size(0), sample_count, replace=False)
    selected_fps = subset[selected_index]

    return evaluate_memory(model, obs, selected_fps, threshold=threshold)

def memory_analysis(variant, ckpt, threshold=10, sample_count=50, seed=0):
    # Grab number of memory eigenvalues (time const > `threshold`) for all beliefs
    df = load_variant(variant, ckpt)
    labels = get_labels(variant, ckpt)

    past_thresh = []
    for index in range(6):
        past_thresh.append(
            memory_analysis_single(
                variant, ckpt, index,
                threshold=threshold,
                sample_count=sample_count,
                seed=seed,
                df=df
            )
        )
        print(labels[index], past_thresh[-1])
    return labels, past_thresh


#%%
# Ranking plot
variants = ['base4-full', 'no_gid-full', 'base-full', 'no_cp-full', 'split_clamp-full', 'no_adp-full']
ckpts = [34, 34, 31, 34, 31, 34]

threshold = 100
memory_info = {}
for variant, ckpt in zip(variants, ckpts):
    memory_info[(variant, ckpt)] = memory_analysis(variant, ckpt, threshold=threshold)
    # memory_info[(variant, ckpt)] = memory_analysis(variant, ckpt)

torch.save(memory_info, f'memory_rank_cache_{threshold}.pth')

#%%

# memory_info = torch.load(f'memory_rank_cache_{threshold}.pth')
memory_info = torch.load(f'memory_rank_cache.pth')
# Cache it
# print(len(memory_info))
#%%
# We have a statistic for each model/ckpt/belief. How do we crunch it down?
orders = ['CPC|A-4', 'CPC|A-16', 'ADP', 'PBL', 'Base', 'GID', 'CP']
# Force a specific order (to insert into DF => plot order)
indices = [0, 2, 5, 1, 6, 3, 4]

df_info = []
for key in memory_info:
    labels, sample = memory_info[key]
    ranks = stats.rankdata(sample).astype(int)
    for label, rank, memory_size in zip(labels, ranks, sample): # a bunch of numbers, basically
    # for i in indices:
        # label, rank, memory_size = labels[i], ranks[i], sample[i]
        order = orders.index(label)
        df_info.append({
            'variant': sample[0],
            'ckpt': sample[1],
            'memory': memory_size,
            'rank': rank,
            'belief': label,
            'order': order
        })

# I have a hunch it's going to be sorted in the order it's added
# Swap it out if not
for i, order in enumerate(orders):
    if df_info[i]['belief'] == order:
        continue
    # first item
    for swap_index in range(len(df_info) - 1, i, -1):
        if df_info[swap_index]['belief'] == order:
            swap = df_info[swap_index]
            df_info[swap_index] = df_info[i]
            df_info[i] = swap
            break


df = pd.DataFrame(df_info)

fig = plt.figure()
ax = fig.gca()
prep_plt(ax=ax)
colormap = sns.color_palette(palette='flare', n_colors=6, as_cmap=True)

ax = sns.histplot(data=df, x="belief", hue='rank', multiple='stack', ax=ax, palette=colormap, alpha=0.9)
# ax.set_xlabel('Belief Task')
ax.text(2., -1.6, 'Belief Task', size=16)
ax.set_xlabel('')
ax.set_yticks([0, 3, 6])
ax.set_yticklabels([0, 3, 6], size=14)
ax.set_xticklabels(orders, rotation=30, size=14) #, horizontalalignment='right')

import matplotlib.cm as cm
import matplotlib
# normalize = mcolors.Normalize
# norm = matplotlib.colors.Normalize(vmin=0,vmax=6)
# norm = mpl.colors.BoundaryNorm(np.arange(7) + 1, colormap.N)
# scalarmappable = cm.ScalarMappable(norm=norm, cmap=colormap)
# scalarmappable.set_array(range(6))
# cbar = ax.figure.colorbar(scalarmappable, ax=ax, orientation='horizontal')
# cbar.ax.set_yticks(np.arange(6) + 1)
# cbar.ax.set_yticklabels(np.arange(6))
# plt.colorbar(orientation='h', cax=ax)

# fig.colorbar(scalarmappable, cax=ax)
sns.despine(ax=ax)
# cmap2 = sns.color_palette(palette='flare', n_colors=6, a=0.8)
inset_ax = fig.add_axes([0.6, 0.8, 0.3, 0.05])
n = 6
inset_ax.imshow(np.arange(n).reshape(1, n),
            # cmap=mpl.colors.ListedColormap(cmap2),
            cmap=colormap, # mpl.colors.ListedColormap(list(colormap)),
            interpolation="nearest", aspect="auto")
inset_ax.set_xticks(np.arange(n)[::-1])
inset_ax.set_xticklabels(np.arange(n) + 1)
inset_ax.set_yticks([-.5, .5])
# inset_ax.set_xticklabels(["" for _ in range(n)])
inset_ax.set_yticks([])
inset_ax.text(0.1, -0.7, 'Memory Rank', size=16)

ax.grid(axis='x')
ax.get_legend().remove()

# legend = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)][0]
# legend.set_title("Memory Rank")
# legend.set_bbox_to_anchor((1.01, 1.0))

fig.savefig('test.pdf', dpi=150, bbox_inches="tight")
# TODO do with 100

#%%
# Increase over time
threshold = 100 # 10 is too easy
variants = ['base4-full'] * 7 + ['base-full'] * 7 + ['split_clamp-full'] * 7

ckpts = [
    4, 9, 13, 19, 25, 31, 35,
    4, 9, 13, 19, 25, 31, 35,
    4, 9, 13, 19, 25, 31, 35,
]

memory_info = {}
# for belief in [4]:
for belief in [0, 4]:
    for var, ckpt in zip(variants, ckpts):
        # memory_info[(var, ckpt, belief)] = memory_analysis_single(var, ckpt, index=belief)
        memory_info[(var, ckpt, belief)] = memory_analysis_single(var, ckpt, index=belief, threshold=threshold, sample_count=50)

torch.save(memory_info, 'memory_time_cache.pth')
#%%
memory_info = torch.load('memory_time_cache.pth')


#%%
BELIEF_ABBREVIATIONS = {
    "CPCA": "CPC|A-4",
    "CPCA_B": "C16",
    "CoveragePrediction": "CP",
    # "CoveragePrediction": "CovPred",
    "ActionDist_A": "ADP",
    "Dummy": "Base"
}

def get_labels(variant, ckpt):
    ckpt = torch.load(f'/srv/flash1/jye72/share/objectnav/{variant}/{variant}.{ckpt}.pth', map_location='cpu')
    tasks = ckpt['config']['RL']['AUX_TASKS']['tasks']
    return [BELIEF_ABBREVIATIONS.get(t, t) for t in tasks]

VARIANT_LABELS = {
    'base4-full': '4-Act',
    'base-full': '6-Act',
    'split_clamp-full': '6-Act + Tether'
}

palette = sns.color_palette(palette='muted', n_colors=3, desat=0.9)
df_info = []
for key in memory_info:
    variant, ckpt, belief = key
    sample = memory_info[key]
    df_info.append({
        'Variant': f"{VARIANT_LABELS[variant]}",
        'Belief': f"{get_labels(variant, ckpt)[belief]}",
        'ckpt': ckpt,
        'memory': sample,
    })

df = pd.DataFrame(df_info)

# Lineplot -- demonstrating memory appears to saturate,
# and that the exploration objective forces a lot of additional "memory"
prep_plt()
plt.rc('legend', fontsize=11, frameon=True) # , loc=(1.0, 0.5))    # legend fontsize

# g = sns.lineplot(x='ckpt', y='memory', hue='Variant', style='Belief', markers=True, dashes=False, data=df)
g = sns.lineplot(x='ckpt', y='memory', hue='Variant', style='Belief', dashes=False, data=df)
g = sns.scatterplot(x='ckpt', y='memory', hue='Variant', style='Belief', s=100, data=df)
sns.despine(ax=g)

g.set_xticks([12, 23, 34])
g.set_yticks([0, 40, 80, 120, 160])
g.set_yticklabels([0, 40, 80, 120, 160], size=14)

# strs = map(lambda label: label._text, g.get_xticklabels())
# print(list(strs))
# ckpt_to_frames = lambda x: f"{(123 / 34.) * float(x):.1g}"
# g.set_xticklabels(map(ckpt_to_frames, strs), rotation=45, horizontalalignment='right')
g.set_xticklabels([40, 80, 120], size=14)
g.set_xlabel('Training Steps (Million)')
# plt.setp(g.collections, sizes=[100])

# No marker at the moment...

g.set_ylabel('Count of 100+ step modes')
legend = g.legend()
# legend = [c for c in g.get_children() if isinstance(c, mpl.legend.Legend)][0]
# handles = legend.legendHandles[:len(legend.legendHandles) // 2]
handles = legend.legendHandles[1:4] # + legend.legendHandles[-2:]
# legend.legendHandles = legend.legendHandles[:len(legend.legendHandles) // 2]
# legend.texts = legend.texts[:len(legend.texts) // 2]
new_labels = [
    # 'Variant',
    '4-Act', # saturating curve
    '6-Act', # saturating curve
    '6-Act + Tether', # preserved
    # 'Belief',
    # 'CPC|A-4', # task seems to determine memory accumulated
    # 'CP', # task seems to determine memory accumulated
]
for t, l in zip(legend.texts, new_labels): t.set_text(l)
# g.legend(handles, new_labels, loc=(0.50, 0.4), frameon=False, fontsize=14, ncol=2, columnspacing=-3.0)
# g.legend(handles, new_labels, loc=(0.55, 0.43), frameon=False, fontsize=14, markerfirst=False)
g.legend([], frameon=False)

g.text(8, 40, '$\\bullet$ CPC|A-4', size=20, fontweight='bold')
g.text(8, 155, '$\\times$ CP', size=20, fontweight='bold')

g.text(29, 60, "4-Act", size=16, color=handles[0].get_color())
g.text(29, 21, "6-Act", size=16, color=handles[1].get_color())
g.text(22, 6, "6-Act + Tether", size=16, color=handles[2].get_color())

# ax.legend(legend.legendHandles, legend.texts)
# new_labels = [
#     '4-Act C4', # saturating curve
#     '6-Act C4', # saturating curve
#     '6-A Tether C4', # preserved
#     '4-Act CP', # task seems to determine memory accumulated
#     '6-Act CP', # task seems to determine memory accumulated
#     '6-A Tether CP' # drops a lot
# ]
# legend.set_bbox_to_anchor((1.00, 0.6))

plt.savefig('test.pdf', dpi=150, bbox_inches="tight")


#%%
legend.__dict__.keys()
# print(len(legend.texts))