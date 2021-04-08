#%%
# Produces figures re: curvature + action entropy
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
from torch.distributions import Categorical

import seaborn as sns
import PIL.Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from analyze_utils import (
    prep_plt, plot_to_axis, get_belief,
    load_device,
    loc_part_ratio, part_ratio, svd, pearsonr
)

from fp_finder import (
    load_variant, load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path,
    get_jac_pt
)

from fp_plotter import (
    get_pca_view,
    scatter_pca_points_with_labels,
    add_pca_inset,
    plot_spectrum
)

CUDA_DEVICE = 1 # From alloc-ed item
device = load_device(CUDA_DEVICE)

VARIANT_LABELS = {
    # 'base-full': "6-act",
    'base-full': "Base",
    # 'split_clamp-full': "6-act + Tether",
    'split_clamp-full': "Tether",
    'base4-full': '4-act',
    'base_rednet-full': 'Base (RedNet FT)',
    'split_rednet-full': 'Tether (RedNet FT)'
}

variant = 'base-full'
# ckpt = 35
ckpt = 31
# ckpt = 25
# ckpt = 19
# ckpt = 13
# 31, 25, 19, 13

# TODO supplement figure of action ent decrease given RedNet tuning
# variant = 'base_rednet-full' # TODO
# ckpt = 37

is_eval = True
# is_eval = False
is_gt = False
is_gt = True

def get_belief_attr(df, scene_key, ep_id, index=0, key='fused_belief'):
    row = df[(df['scene'] == scene_key) & (df['ep'] == ep_id)]
    el = row[key].values[0]
    return el[:, index] if key == 'beliefs' else el

def get_behavioral_entropy(traj, suffix=0):
    # traj: T x [K] - sequence of actions
    def get_entropy(seq):
        out, counts = torch.unique(seq, return_counts=True)
        probs = counts / counts.sum()
        return Categorical(probs=probs).entropy()
    if len(traj.size()) == 1:
        return get_entropy(traj[suffix:])
        # return get_entropy(traj[suffix:]).item()
    return torch.stack([get_entropy(single_traj[suffix:]) for single_traj in traj.T])

def angle_bn(vector_1, vector_2): # each T x H
    unit_vector_1 = vector_1 / torch.linalg.norm(vector_1, dim=-1, keepdim=True)
    unit_vector_2 = vector_2 / torch.linalg.norm(vector_2, dim=-1, keepdim=True)
    dot_product = (unit_vector_1 * unit_vector_2).sum(dim=-1)
    return torch.arccos(dot_product) / (2 * math.pi) * 360

def get_discrete_curvature(traj, index=-1):
    # traj: T x [B] x [K] x H
    # averages over dim 0, angle on last dim. bach on dim 1
    if index >= 0:
        traj = traj[..., index, :] # T x [B] x H
    pt_one = traj[:-2]
    pt_two = traj[1:-1]
    pt_three = traj[2:]
    vec_one = pt_two - pt_one
    vec_two = pt_three - pt_two
    angle = angle_bn(vec_one, vec_two) # T [x B]
    if len(angle.size()) > 1:
        means = [row[~torch.isnan(row)].mean() for row in angle.T]
        return torch.stack(means)
    angle = angle[~torch.isnan(angle)]
    return angle.mean(dim=0)

def get_rolling_metric(traj, k=100, fn=get_discrete_curvature, average=False):
    # (Mean) metric on last k steps
    if len(traj) < k:
        res = fn(traj)
        if average:
            return res
        else:
            return res.unsqueeze(0) # [1]
    steps = traj.unfold(dimension=0, size=k, step=1) # Windows x H x Length
    if len(steps.size()) == 4:
        steps = steps.permute(3, 0, 1, 2) # 30 x 100 x 6 x 196?
    elif len(steps.size()) == 3:
        steps = steps.permute(2, 0, 1) # L x W x H
    else:
        steps = steps.permute(1, 0) # L x W
    res = fn(steps)
    if average:
        return res.mean(dim=0).item() # some weird casting issue
    return res

def get_loop(x):
    def mean_visitation(seq):
        return seq.float().mean().item() / len(seq)
    return mean_visitation(x.visit_count_t) and x.steps > 250 and x.collisions < 50

def load_curvature_variant(variant, ckpt, is_gt=True, is_eval=True, k=30):
    df = load_variant(variant, ckpt, is_gt=is_gt, is_eval=is_eval)
    # df['fused_curv'] = df.apply(lambda x: get_discrete_curvature(x.fused_belief).item(), axis=1)
    for i in range(6):
        df[f'bel{i}_curv'] = df.apply(lambda x: get_discrete_curvature(x.beliefs, index=i).item(), axis=1)
    df['vis_curv'] = df.apply(lambda x: get_discrete_curvature(x.fused_obs).item(), axis=1)
    # df['action_ent'] = df.apply(lambda x: get_behavioral_entropy(x.actions).item(), axis=1)
    df['action_ent_k'] = df.apply(lambda x: get_rolling_metric(x.actions, k=k, fn=get_behavioral_entropy, average=True), axis=1)
    df['action_ent_k'] = df['action_ent_k'].astype(np.float)
    df['variant'] = VARIANT_LABELS[variant]
    df['success_label'] = df.apply(lambda x: 'Success' if x.success else 'Failure', axis=1)
    df['Semantics'] = "GT" if is_gt else "Pred"
    # df['action_ent_k'] = df.apply(lambda x: get_rolling_metric(x.actions, k=k, fn=get_behavioral_entropy), axis=1)
    # df['loop'] = df.apply(get_loop, axis=1)
    # df['fused_curv_k'] = df.apply(lambda x: get_rolling_metric(x.fused_belief, k=k, fn=get_discrete_curvature), axis=1)
    # for i in range(6):
    #     df[f'bel{i}_curv_k'] = df.apply(lambda x: get_rolling_metric(x.beliefs, k=k, fn=lambda t: get_discrete_curvature(t, index=i)), axis=1)
    # df['vis_curv_k'] = df.apply(lambda x: get_rolling_metric(x.fused_obs, k=k, fn=get_discrete_curvature), axis=1)
    return df

df = load_curvature_variant(variant, ckpt, is_eval=is_eval, is_gt=is_gt)
# df.set_index(['scene', 'ep'])
#%%
df1 = load_curvature_variant('base_rednet-full', 36, is_eval=True, is_gt=False)
df2 = load_curvature_variant('split_rednet-full', 37, is_eval=True, is_gt=False)
df3 = load_curvature_variant('base-full', 31, is_eval=True, is_gt=False)
df4 = load_curvature_variant('split_clamp-full', 31, is_eval=True, is_gt=False)

#%%
df1['variant'] = 'Base'
df2['variant'] = 'Tether'
df1['Tuned'] = True
df2['Tuned'] = True
df3['Tuned'] = False
df4['Tuned'] = False
df = pd.concat([df1, df2, df3, df4])
#%%
# Tuning seems to shift action entropy up. We suspect that action entropy isn't really the measure we want.
g = sns.catplot(x="variant", y="action_ent_k",
    # hue="success",
    hue="Tuned", col="success",
    # hue="Semantics", col="success_label",
    # hue="success", col="is_gt",
    data=df, kind="violin", split=True,
    # data=df, kind="violin", split=True,
    height=4, aspect=.7)

#%%
# sns.scatterplot(x='vis_curv', y='action_ent_k', data=df)
ax = sns.violinplot(x='success', y='action_ent_k', split=True, data=df)

#%%
pairs = [
    # ('base4-full', 35, True), ('base4-full', 35, False),
    ('base-full', 31, True), ('base-full', 31, False),
    ('split_clamp-full', 31, True), ('split_clamp-full', 31, False),
]
dfs = [load_curvature_variant(variant, ckpt, is_eval=False, is_gt=is_gt) for \
    (variant, ckpt, is_gt) in pairs
]
df = pd.concat(dfs)

#%%
# Measure correlation between curvature and obs.

df['fused_act_corr'] = df.apply(lambda x: pearsonr(x.action_ent_k, x.fused_curv_k).item(), axis=1)
df['obs_act_corr'] = df.apply(lambda x: pearsonr(x.action_ent_k, x.vis_curv_k).item(), axis=1)

for i in range(6):
    df[f'bel{i}_act_corr'] = df.apply(lambda x: pearsonr(x.action_ent_k, x[f'bel{i}_curv_k']).item(), axis=1)

all_corrs = []
temp_df = df[[f'obs_act_corr']].copy()
temp_df.columns = ['corr']
temp_df['source'] = f'obs_act_corr'
all_corrs.append(temp_df)
temp_df = df[[f'fused_act_corr']].copy()
temp_df.columns = ['corr']
temp_df['source'] = f'fused_act_corr'
all_corrs.append(temp_df)
for i in range(6):
    temp_df = df[[f'bel{i}_act_corr']].copy()
    temp_df.columns = ['corr']
    temp_df['source'] = f'bel{i}_act_corr'
    all_corrs.append(temp_df)

# palette = sns.color_palette(palette='muted', n_colors=len(all_corrs), desat=0.3)

corr_df = pd.concat(all_corrs)
ax = sns.kdeplot(x='corr', hue='source', data=corr_df, cut=0)
# sns.histplot(x='corr', hue='source', data=corr_df, kde=True) #, palette=palette)
sns.despine(ax=ax)
ax.set_title(variant)

#%%
# OBSERVATION CORRELATION
# We do this to show that hidden states are heavily tied to agent trajectory (behavior and inputs), as opposed to just doing their own thing. They could still be doing their own thing, but it's on top of working with the inputs.

BELIEF_LABELS = ['CPC|A-4', 'PBL', 'CPC|A-16', 'GID', 'CP', 'ADP']

df['fused_obs_corr'] = df.apply(lambda x: pearsonr(x.vis_curv_k, x.fused_curv_k).item(), axis=1)
for i in range(6):
    df[f'bel{i}_obs_corr'] = df.apply(lambda x: pearsonr(x.vis_curv_k, x[f'bel{i}_curv_k']).item(), axis=1)

all_corrs = []
temp_df = df[[f'fused_obs_corr']].copy()
temp_df.columns = ['corr']
temp_df['Belief'] = 'Fused'
# temp_df['source'] = f'fused_obs_corr'
all_corrs.append(temp_df)
for i in range(6):
    temp_df = df[[f'bel{i}_obs_corr']].copy()
    temp_df.columns = ['corr']
    temp_df['Belief'] = BELIEF_LABELS[i] # f'bel{i}_obs_corr'
    all_corrs.append(temp_df)

corr_df = pd.concat(all_corrs)
prep_plt()
plt.rc('legend', fontsize=14, frameon=True, title_fontsize=16) # , loc=(1.0, 0.5))    # legend fontsize

ax = sns.kdeplot(x='corr', hue='Belief', data=corr_df, cut=0)
legend = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)][0]
line_handles = legend.legendHandles[1:] # + legend.legendHandles[-2:]

legend.remove()

ax.text(0.45, 0.17, "CP", color=line_handles[-2].get_color(), size=20, fontweight='bold')
ax.text(0.78, 0.33, "Other\nBeliefs", size=20, horizontalalignment='right')
sns.despine(ax=ax)
ax.set_title("")
ax.set_xlabel('Observation Correlation $r$')
ax.set_yticks([0.0, 0.2, 0.4])
# ax.set_yticks([0.0, 0.2, 0.4])
plt.savefig('test.pdf', bbox_inches='tight')


#%%
# View single correlations hued by success

# Single instances
# sort_df = df[df['success_label'] == 'Failure'].sort_values(by=['fused_act_corr'])
# print(sort_df.iloc[0][['fused_act_corr', 'scene', 'ep']].values)
# print(sort_df.iloc[1][['fused_act_corr', 'scene', 'ep']].values)
# print(sort_df.iloc[2][['fused_act_corr', 'scene', 'ep']].values)
# print(sort_df.iloc[3][['fused_act_corr', 'scene', 'ep']].values)

sns.histplot(x='fused_act_corr', hue='success_label', data=df)
sns.histplot(x='obs_act_corr', hue='success_label', data=df)

"""
How does this motivation sound:
I propose that action entropy and curvature are very correlated. this is empirically vetted. It happens either because action entropy drives vision entropy drives curvature, or because RNNs control actions. Either way, they're correlated.
but THUS, when they're NOT correlated, some interesting computation must be happening?
"""
#%%

# ilocs = range(10)
# sns.lineplot(y='fused_act_corr', data=dfs)
scene_key = '2azQ1b91cZZ'
scene_key = '8194nk5LbLH'
ep_id = 14
scene_key = 'TbHJrupSAjP'
ep_id= 21
# for i in ilocs:
    # scene_key, ep_id = df.iloc[i][['scene', 'ep']].values
    # scene_key, ep_id = df.iloc[i].values
plot_series(df, scene_key, ep_id, k=30)


# ax = sns.violinplot(x='variant', y='action_ent', hue='success', split=True, data=df)

#%%
prep_plt()
# sns.scatterplot(x='vis_curv', y='collisions', hue='success', data=df)
# sns.scatterplot(x='fused_curv', y='collisions', hue='success', data=df)
# sns.scatterplot(x='bel4_curv', y='coverage', hue='success', data=df)
# sns.scatterplot(x='bel0_curv', y='coverage', hue='success', data=df)
# ax = sns.scatterplot(x='action_ent', y='coverage', hue='success', data=df)
# ax = sns.scatterplot(x='action_ent', y=1.0, hue='success', data=df)
# ax = sns.scatterplot(x='action_ent_suffix', y=, hue='success', data=sdf)

# ax = sns.violinplot(x='variant', y='action_ent', hue='is_gt', split=True, data=df)
g = sns.catplot(x="variant", y="action_ent_k",
    # hue="success",
    hue="Semantics", col="success_label",
    # hue="success", col="is_gt",
    data=df, kind="violin", split=True,
    height=4, aspect=.7)

(g.set_axis_labels("", "10-step Action Entropy")
    .set_titles("{col_name}")
    )

# ax = sns.histplot(x='action_ent', hue='success', bins=10, data=df)

# sns.scatterplot(x='action_ent', y='scene', hue='success', data=df)

# sns.scatterplot(x='fused_curv', y='scene', hue='success', data=df)

# sns.scatterplot(x='fused_curv', y='coverage', hue='success', data=df)

# sns.histplot(x='fused_curv', hue='success', data=df) # ??? kde=True doesn't work

# plt.title(f"{VARIANT_LABELS[variant]} {'GT' if is_gt else 'Pred'} {'Val' if is_eval else 'Train'}")
# plt.title(f"{variant} {ckpt} {'gt' if is_gt else 'pred'} {'eval' if is_eval else 'train'}")

g.savefig('test.pdf')

#%%
scene_key = 'QUCTc6BB5sX'
ep_id = 10 # looping
# ep_id = 8 # terminal confusion

# TODO other metrics

# scene_key = 'Z6MFQCViBuw'
# ep_id = 13

# scene_key = '8194nk5LbLH'
# ep_id = 17

# scene_key = '2azQ1b91cZZ'
# ep_id = 0

scene_key = 'zsNo4HB9uLZ'
ep_id = 15
# scene_key = 'Vvot9Ly1tCj'
# ep_id = 2

# Hmm, curvature increases despite relatively stable exploration

traj = get_belief_attr(df, scene_key, ep_id)
# traj = get_belief_attr(df, scene_key, ep_id, index=0)


def plot_series(df, scene_key, ep_id, **kwargs):
    prep_plt()
    traj = get_belief_attr(df, scene_key, ep_id)
    plt.plot(get_rolling_metric(traj, **kwargs) / 90, label='fused')
    # for i in range(6):
    #     traj = get_belief_attr(df, scene_key, ep_id, key='beliefs', index=i)
    #     plt.plot(get_rolling_metric(traj, **kwargs) / 90, label=i)

    traj = get_belief_attr(df, scene_key, ep_id, key='fused_obs')
    plt.plot(get_rolling_metric(traj, **kwargs) / 90, label='obs')

    traj = get_belief_attr(df, scene_key, ep_id, key='actions')
    plt.plot(get_rolling_metric(traj, fn=get_behavioral_entropy, **kwargs), label='action')

    plt.legend(frameon=False)
    plt.title(f"{variant} {ckpt} {scene_key} Ep {ep_id}")

# plot_series(df, scene_key, ep_id, k=50) # Wow, that correlation
# plot_series(df, scene_key, ep_id, k=10)
plot_series(df, scene_key, ep_id, k=30)

# plot_series(df, scene_key, ep_id, k=10)
# ! Correlation is probably causal in the other direction?
# We want to show that behavioral is tied to hidden state trajectory
# We know this is true in the forward direction. I guess this is to show it goes full circle.
# In a quantitative way.


# plot_series(df, scene_key, ep_id) # Wow, that correlation

#%%
# row = df[(df['scene'] == scene_key) & (df['ep'] == ep_id)]

scene_key = 'oLBMNvg9in8'
ep_id = 13
plot_series(df, scene_key, ep_id=ep_id, k=30)


#%%
# print(row)
# failures = df[df['success'] == 0][['scene', 'ep']]
# failures = df[df['success'] == 1][['scene', 'ep']]

ilocs = range(5)
ilocs = []
for i in ilocs:
    # scene_key, ep_id = failures.iloc[i].values
    scene_key, ep_id = df.iloc[i].values
    plot_series(df, scene_key, ep_id, k=30)


#%%


# plot_series(df, scene_key, ep_id, fn=get_behavioral_entropy)

# plot_series(df, scene_key, ep_id, fn=get_discrete_curvature)
traj = get_belief_attr(df, scene_key, ep_id, key='actions')
plt.plot(get_rolling_metric(traj, fn=get_behavioral_entropy))
# traj = get_belief_attr(df, scene_key, ep_id)
# plt.plot(get_rolling_metric(traj, fn=get_discrete_curvature))
