#%%
# Notebook for analyzing fixed points (used for qualitative examples in appendix)

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

variant = "split-curric"
ckpt = 31
# ckpt = 37

# variant = "split_120-curric"
# ckpt = 26
# variant = "feed-curpol"
# ckpt = 57

# variant = 'base4-full'
ckpt = 35
variant = 'base-full'
# ckpt = 35
ckpt = 31
# ckpt = 25
# ckpt = 19
# ckpt = 13

# variant = 'base4-full'
# variant = 'no_cp-full'
# ckpt = 34
# ckpt = 24
# ckpt = 19
# ckpt = 14

variant = 'split_clamp-full'
ckpt = 31
# ckpt = 35
# fp_tag = "zero"
fp_tag = "nojitter"
fp_tag = ""

seed = 0
num_fp = 10000
scene = 'all'
# index = 0
# index = 4
index = 0
index = 1
# index = 2
# index = 3
index = 4
# index = 5

df = load_variant(variant, ckpt)

tag = f"{variant} {ckpt} Bel. {index}"
override_cache = True
override_cache = False

if scene == 'all':
    conditions = df
else:
    conditions = df[df['scene'] == scene]

model = load_recurrent_model(variant, ckpt, index=index)
obs, *_ = get_model_inputs(
    df, scene=scene, index=index, size=num_fp, seed=seed
)
mean_obs = obs

cache_fps = get_cache_path(
    variant, ckpt, num_fp, index, scene, tag=fp_tag
)
print(cache_fps)
if override_cache or not osp.exists(cache_fps):
    run_fp_finder(
        variant=variant, ckpt=ckpt, index=index,
        num_fp=num_fp, scene=scene,  seed=seed
    )
info = torch.load(cache_fps, map_location='cpu')
fps = info['fps']
ids = info['ids']
choices = info['choices']
deltas = info['deltas']

pca = PCA(n_components=30)
pca.fit(fps)

label = deltas > 4e-7
plt.scatter(deltas[~label], np.random.rand(*deltas[~label].shape), s=3)
plt.scatter(deltas[label], np.random.rand(*deltas[label].shape), s=3)
plt.xlim(0, 1e-5)
print(label.size())
# * ids log the df index of each FP/its source hidden state episode

# fps.size()
# plt.plot(fps.std(-1))
# Hmm, there really only seems to be one fixed point for C4
#%%
print(fps.size())
plt.plot(fps.std(0))


#%%
# * Inspect points
view_indices = [0, 1]
# view_indices = [2, 3]
# view_indices = [0, 1, 2]
sample_traj_inds = [0, 10, 20, 30] #, 1, 2]
# sample_traj_inds = [0] # , 10, 20, 30] #, 1, 2]
# sample_traj_inds = [] # , 10, 20, 30] #, 1, 2]
# sample_traj_inds = [0, 1, 2, 3] #, 1, 2]
fig = plt.figure()

def plot_trajectories(pca, traj_inds, ax, view_indices=[0,1], colorbar=False, palette=None, **kwargs):
    # * Global 'df'
    samples = [get_belief(conditions, i, **kwargs) for i in traj_inds]
    reduced_trajs = [pca.transform(s)[:, view_indices] for s in samples]
    for r in reduced_trajs:
        plot_to_axis(fig, ax, r, scatter=True, colorbar=colorbar, palette=palette)

def plot_fixed_points(
    fig, points, pca, traj_inds=[], ax=None, view_indices=[0, 1], key=None, **kwargs
):
    if index == 0:
        kwargs['s'] = 30 # emphasize attractor
    ax, _, vals, colors = scatter_pca_points_with_labels(
        fig,
        points,
        pca=pca,
        ax=ax,
        view_indices=view_indices,
        key=key,
        **kwargs
    )
    if colors is not None:
        print(len(colors))
        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, vals)]
        # ax.legend(handles=patches, frameon=False, loc=(1.01, 0.2), title=key)
    # ax.set_title(f"{tag} Fix Pts")
    plot_trajectories(pca, traj_inds, ax, view_indices=view_indices, index=index)
    return ax

loc_parts, lpr = loc_part_ratio(fps, sample_size=50, k=100)
gpr = part_ratio(fps)
ax = plot_fixed_points(fig, fps, pca,
    view_indices=view_indices,
    traj_inds=sample_traj_inds,
    source_df=conditions,
    ilocs=ids,
    key="obj_cat" if scene == "all" else None
    # key="scene" if scene == "all" else None
)

labels = ['CPC|A-4', 'PBL', 'CPC|A-16', 'GID', 'CovPred', 'ADP']
locs = [
    (-0.1, 0.25),
    (-5, 7),
    (5, 10),
    (5, 10),
    (-5, 5),
    (-10, 7),
]

if index == 0:
    ax.text(x=locs[index][0], y=locs[index][1], s=f"{labels[index]}\nSingle Point", ha='center', size=14)
else:
    ax.text(x=locs[index][0], y=locs[index][1], s=f"{labels[index]} | Global PR: {gpr:.2g} Local PR: {lpr:.2g}", size=14)
# ax2 = add_pca_inset(fig, pca)
# ax2.text(x=40, y=0.5, s=f"Global PR: {gpr:.2g} Local PR: {lpr:.2g}")
fig.savefig('test.pdf')

#%%

from fp_finder import extract_flat_key
# index = 'Fused'
# trajs, _ = extract_flat_key(df, 'fused_belief')
# index = 1
# trajs, _ = extract_flat_key(df, 'beliefs', index=index)
# print(trajs.size())
# pca = PCA(n_components=10)
# pca.fit(trajs)

sample_traj_inds = range(20)
sample_traj_inds = [1]
view_indices = [0, 1]
# view_indices = [0, 1, 2]
scene_key = 'Z6MFQCViBuw'
ep_id = 109

scene_key = 'QUCTc6BB5sX'
ep_id = 10

scene_key = '8194nk5LbLH'
ep_id = 19
ep_id = 14

def plot_eps(ep_ids, same_axis=True): # Only supports up to 2 eps
    # fig = plt.figure()
    palettes = [
        sns.color_palette("icefire", as_cmap=True),
        # sns.color_palette("flare", as_cmap=True),
        sns.color_palette("crest", as_cmap=True),
    ]
    if same_axis:
        f = plt.figure()
        if len(view_indices) == 3:
            ax = f.add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
        axes = [ax] * len(ep_ids)
    else:
        f, axes = plt.subplots(nrows=len(ep_ids), ncols=1, figsize=(3, 4))
    for ep_id, ax, pal in zip(ep_ids, axes, palettes[:len(ep_ids)]):
        row = conditions[(conditions['scene'] == scene_key) & (conditions['ep'] == ep_id)]
        sample_traj_inds = row.index
        print(conditions.iloc[sample_traj_inds][['ep', 'scene', 'success']])
        plot_trajectories(
            pca, sample_traj_inds, ax=ax,
            view_indices=view_indices,
            colorbar=len(view_indices) == 2 and (len(ep_ids) == 1 or not same_axis),
            index=index,
            fused=index == 'Fused',
            palette=pal
        )
        ax.set_title(f'{variant} {index} {ep_id}')
    plt.tight_layout()
# plot_eps([94])
plot_eps([ep_id])
# plot_eps([77, 94])

#%%
# Quick curvature check on these bits.
a = get_belief(conditions, 0)
def extract_episode(scene, ep_id, **kwargs):
    row = conditions[(conditions['scene'] == scene_key) & (conditions['ep'] == ep_id)]
    return get_belief(conditions, ep_id=row.index, **kwargs)
samples = extract_episode(scene, ep_id, fused=True)
print(samples.shape)

#%%
# ! Careful, this number can't be big with this inefficient jacobian implementation
sample_count = 50
# Jacobian peeks
np.random.seed(0)

# Threshold test - FAST
is_fast = True
is_fast = False
subset = fps[label if is_fast else ~label]
print(subset.size(0))
selected_index = np.random.choice(subset.size(0), sample_count, replace=False)
selected_fps = subset[selected_index]

indices = [0, 1] # 2d
# * FYI you just ran something with the thresholded, ideally we see even more stable modes.
obs_grad, state_grad = get_jac_pt(model, mean_obs, selected_fps)

#%%
def plot_spectra(ax, state=True):
    prep_plt(ax)
    items = [plot_spectrum(j, ax) for j in (state_grad if state else obs_grad)]
    lines, spectra = zip(*items)

    stable = np.array([(np.abs(s - 1) < 0.01).sum() for s in spectra])
    ax.set_title(f"{tag} {'fast' if is_fast else 'slow'} FP J_{'rec' if state else 'in'} $\sigma$ n={len(lines)}")
    ax.text(
        0.2, 0.7,
        f'Ckpt {ckpt} Stable span: {stable.mean():.3g} $\pm$ {stable.std():.3g}',
        transform=ax.transAxes
    )
    return lines
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
lines = plot_spectra(ax2, state=True)
# lines = plot_spectra(ax2, state=False)

# plot_fixed_points(fig, fps, pca, ax=ax)
# plot_fixed_points(fig, selected_fps, pca,
#     ax=ax, colors=[l.get_color() for l in lines], marker='^')


#%%
vals, vecs = get_eigvals(state_grad)
def plot_vals(spectrum):
    x = [x.real for x in spectrum]
    y = [x.imag for x in spectrum]
    sns.scatterplot(x, y)

print(variant, ckpt, index, (time_const(vals) > 10).sum(axis=-1).mean())

#%%
time_consts = time_const(vals) # B
# print()
prep_plt()
for i in range(10):
    plt.plot(time_const(vals[i]))
    # plt.plot(time_const(vals[i]))
plt.yscale('log')
plt.ylabel('$\\tau$')
plt.xlabel('$\lambda$ (index)')

sns.despine()
plt.savefig('test.pdf', bbox_inches='tight')

#%%
prep_plt()
for i in range(10):
    plot_vals(vals[i])

plt.yticks(np.linspace(-0.4, 0.4, 5))
plt.xlabel('$\mathrm{Re}(\lambda)$')
plt.xlabel('$\mathrm{Im}(\lambda)$')
sns.despine()

plt.savefig('test.pdf', bbox_inches='tight')
#%%
# ids
fig = plt.figure()
sample_id = ids[0]
print(conditions.iloc[sample_id][['ep', 'scene', 'success', 'steps']])

plot_pca_points(fig, fps, pca=pca, ax=plt.gca())
plot_trajectories(1, pca, [sample_id], plt.gca())


#%%
# Prep for sampling
sample_count = 2
seed = 1
np.random.seed(seed)
# selected_index = np.random.choice(fps.size(0), sample_count)
# selected_index = [10]

selected_fps = fps[selected_index]
indices = [0, 1] # 2d

obs_grad, state_grad = get_jac_pt(model, mean_obs, selected_fps)

# Match input jacobians to PC directions
pca = PCA(n_components=196)
pca.fit(fps)
#%%
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121)
plot_fixed_points(fig, fps, pca, ax=ax)
plot_fixed_points(fig, selected_fps, pca, ax=ax, s=20)
ax2 = fig.add_subplot(122)
prep_plt(ax2)

# Is J_rec or J_in
state = True
# state = False

plotted = state_grad if state else obs_grad
jac = plotted[0] # * jac of 1 sampled point
U, S, Vt = svd(jac)

print(U.shape)
print(Vt.shape)

# Show alignment of output vecs against PCs?
def jac_cos_alignment(vec, references):
    return [ref @ vec for ref in references]

# Wait, just do a similarity matrix.

# similarities = pca.components_ @ U # N x H @ U (H x H).
# similarities = pca.components_ @ Vt.T # N x H @ U (H x H).
similarities = U.T @ Vt.T
# Which input dimensions are getting matched to which out dims?

im = ax2.imshow(similarities)
fig.colorbar(im, ax=ax2, shrink=0.6)

ax2.set_xlabel(f"J_{'rec' if state else 'in'} Left SVs")
ax2.set_ylabel(f"FP PCs")
# ax2.set_xlabel(f"J_{'rec' if state else 'in'} Right")
# ax2.set_ylabel(f"J_{'rec' if state else 'in'} Left")

def add_pc_diag_inset(ax):
    sns.despine(ax=ax3)
    # plot_spectrum(plotted[0], ax3)
    # plot_spectrum(plotted[1], ax3)
    plot_spectrum(jac, ax)
    ax.plot(np.diag(similarities), label="diag")
    ax.set_yticks([0, 1])
    ax.legend(frameon=False, loc=(0.5, 0.45))
# ax3 = fig.add_axes([0.3, 0.15, 0.2, 0.2])
# add_pc_diag_inset(ax3)

# Sample PC against all SVs (identify most important modes)
# Conclusion -- no important modes.
# indices = [0, 60, 120, 180]
# sampled = pca.components_
# references = U.T

# for index in indices:
#     ax2.scatter(
#         jac_cos_alignment(pca.components_[], U.T),
#         label=f"PC {pc}", s=3
#     )

# # Converse: Sample SVs against PCs (identify most important PCs)
# for sv in indices:
#     ax2.scatter(
#         np.arange(len(pca.components_)),
#         jac_cos_alignment(U.T[sv], pca.components_),
#         label=f"SV {sv}", s=3
#     )
# ax2.legend(frameon=True, loc=(-0.5, 0))

# TODO Find the PC that the jac output is most aligned with.
# Opts
# 1 -- things are well aligned globally
# 2 -- Jac aligns with local manifold we don't know about
# 3 -- Every hidden jac projects into arbitrary directions, no interp struc


# 196 x 196, 196 x 196, 196 x 233.
# Vt Hidden x Input. Vt[0] describes most sensitive input dim
# U[:, 0] is top output map, (columns)
# Vt @ Vt[0] will give us a vector, but don't think too much about that vector IS, it's just
# in the operating table space (stretchy area).
# print(Vt.shape) # U describes where is projected,
# ax2.plot(S)
# sns.despine(ax=ax2, bottom=True)
# ax2.set_title(f"Cos Similarity Heatmap")

# ax2.set_title(f"Cos Sim. Top PCs vs J_{'rec' if state else 'in'} Left SV")


#%%
from scipy.linalg import subspace_angles

# Do multiple FPs share common jac directions?
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121)
plot_fixed_points(fig, fps, pca, ax=ax)
plot_fixed_points(fig, selected_fps, pca, ax=ax)
ax2 = fig.add_subplot(122)
prep_plt(ax2)

state = True
# state = False

plotted = state_grad if state else obs_grad

U, S, Vt = svd(plotted[0])
U2, S2, Vt2 = svd(plotted[1])

all_choices = np.arange(U.shape[0])

# we can probably divide these spectra into three parts and make a confusion matrix
# of subspace angle AuC. At risk of being too unintrepretable
# We are essentially showing that the different parts of spectra on each jac
# are consistent across FPs

# Mostly aligned
start = 75
slice_size = 100

# Mostly aligned
start = 25
slice_size = 50

# Mostly aligned
start = 0
slice_size = 15

# Custom test -- maybe are the start and end accidentally aligned?

overlap_snippet = np.arange(start, start + slice_size)

random_snippet = np.random.choice(all_choices, size=slice_size, replace=False)

test_overlap = np.rad2deg(subspace_angles(U[:, overlap_snippet], U2[:, overlap_snippet]))
random_overlap = np.rad2deg(subspace_angles(U[:, random_snippet], U2[:, random_snippet]))
ax2.plot(test_overlap, label='test')
ax2.plot(random_overlap, label='random')
ax2.legend(frameon=False)
ax2.set_title('Subspace principal angles')
# We expect
# similarities = U.T @ U2

# im = ax2.imshow(similarities)
# fig.colorbar(im, ax=ax2, shrink=0.6)

#%%
# Distance to fixed point over trajectory -- different view than PC space
# to evaluate how relevant our FPs our

# Close: 0
# Far: 10
def assess_spread(source):
    # Avg norm between FPs and IC is 3-6.
    # What about between ics?
    centroid = source.mean(dim=0)
    source_dists = torch.linalg.norm(source - centroid, dim=-1)
    return source_dists.mean()

obs, ics, *_ = get_model_inputs(
    df, scene=scene, index=index, size=num_fp, seed=seed
)

print(f'avg distance: {assess_spread(ics)}')'

# Analysis "VISUAL RELEVANCE"
# We find that despite appearing trajectories + FPs appearing far in top 2 PC-space,
# the FPs are not really outliers as measured by FPs.
def plot_dist_to_ref(trajs, reference, ax=None):
    # trajs - list of [T, H] hidden states
    # reference - [NxH] reference points.
    if ax is None:
        ax = plt.gca()
    prep_plt(ax)
    reference = torch.tensor(reference)
    dists = [torch.linalg.norm(torch.tensor(traj) - reference, dim=1) for traj in trajs]
    [ax.plot(dist) for dist in dists]

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax2.set_title('Sampled trajectory L2 w/ reference FP')

refs = fps[0:1]
# refs = fps[10:11]
# refs = fps[150:151]
plot_fixed_points(fig, fps, pca, ax=ax)
plot_fixed_points(fig, refs, pca,
    ax=ax, marker='^', s=50)

traj_inds = [0, 1, 2]
samples = [get_belief(conditions, i) for i in traj_inds]
plot_dist_to_ref(samples, refs[0], ax=ax2)
