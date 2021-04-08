#%%
# This notebook checks that our qualitative analysis of failure modes
# Aligns reasonably with heuristic rules for the failure modes.

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
    load_variant, load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path,
    get_jac_pt, get_jac_pt_sequential
)

from fp_plotter import (
    get_pca_view,
    scatter_pca_points_with_labels,
    add_pca_inset,
    plot_spectrum
)

CUDA_DEVICE = 1 # From alloc-ed item
device = load_device(CUDA_DEVICE)

variant = 'base4-full'
ckpt = 34

variant = 'base-full'
ckpt = 31

variant = 'split_clamp-full'
ckpt = 31

is_eval = True
# is_eval = False
is_gt = True
# is_gt = False
# ckpt = 25
# ckpt = 19
# ckpt = 13

# df = load_variant(variant, ckpt, is_eval=is_eval, is_gt=is_gt, override_fn='split_clamp-full/31/train_gt_False_21.pth')
df = load_variant(variant, ckpt, is_eval=is_eval, is_gt=is_gt)
df['collision_time'] = df.apply(lambda x: x.collisions / x.steps, axis=1)
df['cov_time'] = df.apply(lambda x: x.coverage / x.steps, axis=1)
#%%
# Qualify tether degradation
# 1. worse coverage rate/exploration
# 2. spontaneous quits
df = load_variant('split_clamp-full', 31)
df2 = load_variant('base-full', 31)
df['variant'] = 'Tether'
df2['variant'] = 'Base'
df_both = pd.concat([df, df2])
df_both['cov_time'] = df_both.apply(lambda x: x.coverage / x.steps, axis=1)
#%%
# sns.histplot(x='cov_time', hue='success', data=df, binrange=(0, 0.15), bins=15)
palette = sns.color_palette(n_colors=2)
df_both = df_both[df_both['cov_time'] < 0.4] # remove single outlier
prep_plt()
ax = sns.violinplot(x='success', y='cov_time', hue='variant', split=True, inner='quartile', data=df_both, binrange=(0, 0.15), bins=15, palette=palette)
sns.despine(ax=ax)
ax.set_xlabel('')
ax.set_xticklabels(['Failure', 'Success'], fontsize=18)
ax.set_ylabel('Coverage / Time')
ax.legend([], frameon=False)
ax.text(0.6, 0.1, 'Tether', fontsize=20, color=palette[0])
ax.text(1.1, 0.1, 'Base', fontsize=20, color=palette[1])
plt.savefig('test.pdf', bbox_inches='tight')
#%%
# I mean, cool
import torch.nn.functional as F

def smooth(data, window_width=3):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

prep_plt()
sub_df = df[df['success'] == 0]
sub_df2 = df2[df2['success'] == 0]
inds = range(100)
palette = sns.color_palette(n_colors=2)

noted_objs = {}
for i in inds:
    vals = sub_df.iloc[i].critic_values
    vals2 = sub_df2.iloc[i].critic_values
    # acts = sub_df.iloc[i].action_logits.float()
    # acts = F.softmax(acts, dim=-1)
    goal = sub_df.iloc[i].obj_cat
    # if goal not in noted_objs:
    #     noted_objs[goal] = len(noted_objs)
    #     plt.plot(vals[:,0], color=palette[noted_objs[goal]], linestyle='--', alpha=0.5, label=goal)
    # else:
    #     plt.plot(vals[:,0], color=palette[noted_objs[goal]], linestyle='--', alpha=0.5)
    if i == 0:
        # plt.plot(smooth(vals[:,0]), color=palette[0], linestyle='--', alpha=0.2, label=sub_df.iloc[i].obj_cat)
        plt.plot(vals[:,0], color=palette[0], linestyle='--', alpha=0.2, label='tether')
        plt.plot(vals2[:,0], color=palette[1], linestyle='--', alpha=0.2, label='base')
    else:
        plt.plot(vals[:,0], color=palette[0], linestyle='--', alpha=0.2, label='_tether')
        plt.plot(vals2[:,0], color=palette[1], linestyle='--', alpha=0.2, label='_base')
    # plt.plot(acts[:, 0], color=palette[0], linestyle='--', alpha=0.4) # stop prob
    # plt.plot(acts[:,1, 0], color=palette[0], linestyle='--') # stop prob
    # plt.plot(acts[:,1, 1], color=palette[i], linestyle='-.') # forward prob

    # plt.plot(acts[-5:,1, 0], color=palette[i], linestyle='--')
    # plt.plot(vals[-5:,1], color=palette[i])
plt.xlabel("Timestep")
plt.ylabel("Value Prediction")
plt.text(180, 2.5, 'Base', color=palette[1], fontsize=20)
plt.text(0, -0.2, 'Tether', color=palette[0], fontsize=20)
sns.despine(ax=plt.gca())
# plt.legend(loc=(1.01, 0.0))
# plt.title(f"{variant} value")
# plt.title(f"{variant} value")
plt.savefig('test.pdf', bbox_inches='tight')

#%%
# Take a look at full turns. Huh. Looks like base4 does orbit a lot..
LEFT = 2
RIGHT = 3
THRESH = 5
def long_turns(actions, key, thresh=5):
    seq = actions == key
    num = 0
    i = 0
    j = 0
    while i < len(seq) - thresh:
        if all(seq[i:i+thresh]):
            num += 1
            j = i+thresh
        else:
            j = i+1
        while j < len(seq) and not seq[j]:
            j += 1
        i = j
    return num

df['left_turns'] = df.apply(lambda x: long_turns(x.actions, LEFT, thresh=10), axis=1)
df['right_turns'] = df.apply(lambda x: long_turns(x.actions, RIGHT, thresh=10), axis=1)

print('Lefts :', df['left_turns'].sum(), 'Rights :', df['right_turns'].sum())

#%%
def plateau_collisions(coverage, collisions, plateau_thresh=3):
    # coverage - T, collisions - T
    # Find longest plateau and count number of collisions.
    best_plateau_start = 0
    best_plateau_end = 0
    cur_plateaus = [[0, 0, 0]] # start, end, val
    for i in range(len(coverage)):
        next_plateaus = []
        if i > 0 and coverage[i] > coverage[i-1]:
            for p_start, p_end, p_val in cur_plateaus:
                if coverage[i] > p_val + plateau_thresh: # end plateau
                    if p_end - p_start > best_plateau_end - best_plateau_start:
                        best_plateau_start = p_start
                        best_plateau_end = p_end
                else:
                    next_plateaus.append([p_start, i, p_val])
            cur_plateaus = next_plateaus
            cur_plateaus.append([i, i, coverage[i].item()])
        else:
            for p in cur_plateaus:
                p[1] += 1
    for p_start, p_end, p_val in cur_plateaus:
        if p_end - p_start > best_plateau_end - best_plateau_start:
            best_plateau_start = p_start
            best_plateau_end = p_end
    # Get number of collisions in that plateau
    is_collision = collisions[1:] - collisions[:-1]
    is_collision = torch.cat([torch.zeros(1), is_collision], dim=0)
    return sum(is_collision[best_plateau_start:best_plateau_end])
    # You might want to just threshold this e.g.

def bin_peak_collisions(collisions, bin_width=50):
    bins = []
    for i in range(len(collisions) - bin_width):
        bins.append(collisions[i + bin_width] - collisions[i])
    if len(bins) == 0:
        bins.append(collisions[-1])
    return max(bins)

df['plateau_collisions'] = df.apply(
    lambda x: plateau_collisions(x.coverage_t, x.collisions_t, plateau_thresh=2),
    axis=1
)

df['plateau_collisions_ratio'] = df.apply(
    lambda x: x.plateau_collisions / x.steps, axis=1
)

df['binned_collisions'] = df.apply(
    lambda x: bin_peak_collisions(x.collisions_t), axis=1
)

#%%s
# * Debris Obstacle (Plateau collisions)
# Distinct obstacle which prevented further coverage
# 21 failure / 30 total (70%)
# 19 / 21 precision (90%) (2 involved large scale loops)

sns.scatterplot(x='plateau_collisions', y='coverage', hue='success', data=df)
filt_df = df[(df['plateau_collisions'] > 50)]
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'collisions', 'coverage']].sort_values('scene'))

#%%s
# ! Subsumed
# Debris spawn (9/9)
sns.scatterplot(x='collisions', y='coverage', hue='success', data=df)
# sns.scatterplot(x='collision_time', y='coverage', hue='success', data=df)
filt_df = df[(df['collisions'] > 100) & (df['coverage'] < 4)]
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'collisions', 'coverage']])

#%%
# ! Bad.
# Debris obstacle ratio -- this doesn't work because short episodes only a few collisions are needed to flag a short episode
# 16 / 41..
sns.scatterplot(x='plateau_collisions_ratio', y='coverage', hue='success', data=df)
filt_df = df[(df.plateau_collisions_ratio > 0.25)]
print(len(filt_df))
print(len(filt_df[filt_df['success'] == 0][['scene', 'ep', 'collisions', 'coverage']].sort_values('scene')))

#%%
# ! Bad.
# Plateau collisions isn't perfect -- we'd rather track when the agent was stuck in the most recent GPS point
# But binned collisions is even less precise because of constant failures at the end.
sns.scatterplot(x='binned_collisions', y='coverage', hue='success', data=df)

#%%
# ! Bad.
# Debris - Narrow Passageway (Not plateau):
# 4 failure / 5 total (80%)
# 2/4 precision. This is too low.
# sns.scatterplot(x='plateau_collisions', y='collisions', hue='success', data=df)
filt_df = df[(df['plateau_collisions'] <= 50) & (df['collisions'] > 50)]
# sns.scatterplot(x='plateau_collisions', y='collisions', hue='success', data=df)

print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'collisions', 'plateau_collisions', 'coverage']].sort_values('scene'))

#%%
# * Commitment - early SGE + D2G, but failure.
# Surprisingly orthogonal from last mile.
# 13/14 precision. 8194nk 18 is complicated..
# Pathology of coverage reward, probably.

# Expected to be reduced in tether
def early_op(seq, k=200, op=min):
    return op(seq[:k])

df['best_early_d2g'] = df.apply(lambda x: early_op(x.d2g_t), axis=1)
df['best_early_sge'] = df.apply(lambda x: early_op(x.sge_t, op=max), axis=1)
df['best_sge'] = df.apply(lambda x: max(x.sge_t), axis=1)
df['best_d2g'] = df.apply(lambda x: min(x.d2g_t), axis=1)
sns.scatterplot(x='best_sge', y='obj_cat', hue='success', data=df)

# This filter gets a lot more dataset issues and conflation with last milse
# filt_df = df[(df['best_sge'] > 0.1) & (df['success'] == 0)]

filt_df = df[(df['best_early_d2g'] < 1.0) & (df['best_early_sge'] > 0.1) & (df['success'] == 0)]

print(len(filt_df))
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'obj_cat', 'best_early_sge', 'best_early_d2g']])


#%%
# * Last mile
# 11 - full precision.s
def best_final(seq, k=10):
    return max(seq[-k:])
df['best_final_sge'] = df.apply(lambda x: best_final(x.sge_t), axis=1)
filt_df = df[(df['d2g'] < 1.01) & (df['best_final_sge'] > 0.0)]

# filt_df = df[(df['d2g'] < 1.01) & df['sge'] > 0.0]
ax = sns.scatterplot(x='d2g', y='visit_count', hue='success', data=filt_df)
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'd2g', 'sge']])

#%%
# * Looping via mean visitation -- higher is loopier.
# mean visitation has the desirable property that NxN paths are less loopy as N grows
# visited state entropy doesn't have this property -> nxn is always 1/2 loopy.
# line -> 0.
# 13/13 success/count
# 12/13 precision
def mean_visitation(seq):
    return seq.float().mean().item() / len(seq)
df['mean_visitation'] = df.apply(lambda x: mean_visitation(x.visit_count_t), axis=1)

# We kind of need to discount low coverage (short episodes?) and high collisions...
# How do we filter on an adaptive basis?

# sns.scatterplot(x='steps', y='mean_visitation', hue='success', data=df)
# plt.ylim(0, 0.7)
sns.scatterplot(x='steps', y='coverage', hue='success', data=df)
plt.title(f"{variant}, {ckpt}")

filt_df = df[(df['mean_visitation'] > 0.15) & (df['steps'] > 250) & (df['collisions'] < 50)]
print(len(filt_df))
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'obj_cat', 'mean_visitation']])

# sns.stripplot(x='obj_cat', y='mean_visitation', hue='success', dodge=True, data=df)

#%%
# Early Quitting -- just a mystery. Few collisions.
sns.scatterplot(x='steps', y='coverage', hue='success', data=df)

filt_df = df[(df['steps'] <= 250) & (df['collisions'])]

#%%
# Detection - Moderate best SGE, but no dice
sns.scatterplot(x='best_sge', y='best_d2g', hue='success', data=df)
# sns.scatterplot(x='best_sge', y='steps', hue='success', data=df)
filt_df = df[(df['best_sge'] > 0.02) & (df['best_sge'] <= 0.1)]

print(len(filt_df))
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'obj_cat', 'best_sge']].sort_values('scene'))

#%%
# Explore - No SGE, ever . I mean, this is a bad label...
# Prone to void FPs
sns.scatterplot(x='best_sge', y='best_d2g', hue='success', data=df)
# sns.scatterplot(x='best_sge', y='steps', hue='success', data=df)
filt_df = df[(df['best_sge'] == 0.00)]

print(len(filt_df))
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'obj_cat', 'best_sge']].sort_values('scene'))

#%%
# Open -- coverage > 10, collisions < 10
sns.scatterplot(x='coverage', y='collisions', hue='success', data=df)
filt_df = df[(df['coverage'] >= 10.00) & (df['collisions'] < 10)]
print(len(filt_df))
print(filt_df[filt_df['success'] == 0][['scene', 'ep', 'coverage', 'collisions']].sort_values('scene'))


#%%
# TODO get some precision metrics, and take a look at loop values for explore and see overlap with quit

