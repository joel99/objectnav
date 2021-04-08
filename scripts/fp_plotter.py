# Plotting Utils
import numpy as np
import numpy.linalg as linalg

import torch
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import seaborn as sns

from analyze_utils import (
    prep_plt, plot_to_axis, get_belief,
    load_device,
    loc_part_ratio, part_ratio, svd
)

from fp_finder import (
    load_variant, load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path,
    get_jac_pt
)

def get_pca_view(points, pca=None, view_indices=[0, 1]):
    points = points.cpu()
    if pca is None:
        pca = PCA(n_components=30) # low because we're viz-ing
        pca.fit(points)
    return pca.transform(points)[:, view_indices], pca


def map_row_to_key(df, ids, key='scene'):
    if key is None:
        return ids
    return [df.iloc[i][key] for i in ids]

def scatter_pca_points_with_labels(
    fig,
    points, # [N, H]
    pca=None,
    view_indices=[0, 1],
    ax=None,
    plotted=2000, # Slicing. This is a final step, all args should be in full.
    labels=None, # categorical labels for each point [N]
    palette=None,
    source_df=None,
    ilocs=None,
    key=None,
    s=3,
    **kwargs, # Forward to scatter
):
    # Returns plotted axis, scatter patch, categories plotted and their colors.
    assert len(view_indices) in [2,3], "only 2d or 3d view supported"
    if ax is None:
        if len(view_indices) == 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
    reduced, pca = get_pca_view(points, pca=pca, view_indices=view_indices)
    reduced = reduced[:plotted]

    # If no ids are provided and we can generate them.
    if labels is None and not (source_df is None or ilocs is None or key is None):
        labels = map_row_to_key(source_df, ilocs[:plotted], key=key)

    if labels is None:
        scatter = ax.scatter(*reduced.T, s=s, **kwargs) # Sparser plotting
        return ax, scatter, None, None
    else:
        unique, inverse = np.unique(labels[:plotted], return_inverse=True)
        if palette is None:
            palette = sns.color_palette('hls', n_colors=len(unique))
        colors = [palette[c] for c in inverse]
        scatter = ax.scatter(*reduced.T, color=colors, s=s, **kwargs)

    ax.axis('off')
    return ax, scatter, unique, palette

def add_pca_inset(
    fig,
    pca,
    loc=[0.25, 0.15, 0.2, 0.2],
    ax=None,
):
    if ax is None:
        ax = fig.add_axes(loc)
    ax.plot(np.arange(len(pca.components_)), np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel("# of PCs")
    ax.set_ylabel("VAF")
    ax.set_xticks([0, 3, 20])
    prep_plt(ax)
    sns.despine(ax=ax, right=False, top=False)
    return ax

def load_policy_weights(variant, ckpt):
    # Will automatically use second policy if multipolicy
    pth = f"/srv/share/jye72/objectnav/{variant}/{variant}.{ckpt}.pth"
    weights = torch.load(pth, map_location='cpu')['state_dict']
    # print(weights.keys())
    actor_key = 'actor_critic.action_distribution.linear'
    critic_key = 'actor_critic.critic.fc'
    if 'actor_critic.action_distribution.stack.0.linear.weight' in weights:
        actor_key = 'actor_critic.action_distribution.stack.1.linear'
        critic_key = 'actor_critic.critic.stack.1.fc'
    action_weight = weights[f'{actor_key}.weight'].numpy()
    action_bias = weights[f'{actor_key}.bias'].numpy()
    critic_weight = weights[f'{critic_key}.weight'].numpy()
    critic_bias = weights[f'{critic_key}.bias'].numpy()
    return action_weight, action_bias, critic_weight, critic_bias

def plot_spectrum(jac, ax=None):
    U, S, Vt = svd(jac)
    if ax is None:
        ax = plt.gca()
    return ax.plot(S, label="$\sigma$")[0], np.array(S)

def plot_time_constants(jac, ax=None):
    U, S, Vt = svd(jac)
    S = np.minimum(0.99, np.array(S))
    if ax is None:
        ax = plt.gca()
    ax.plot(np.abs(1 / np.log(np.abs(S))))
