#%%
# Plot semantic embeddings learned by the network
# Primarily for Fig A7
import numpy as np
import numpy.linalg as linalg

import torch
import pandas as pd
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import seaborn as sns
import PIL.Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from analyze_utils import (
    prep_plt, plot_to_axis, get_belief,
    load_device,
    loc_part_ratio, part_ratio, svd,
    task_cat2mpcat40_labels as goal_labels,
    mpcat40_labels,
    load_stats
)

from fp_finder import (
    load_variant, load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path
)
from fp_plotter import (
    get_pca_view,
    scatter_pca_points_with_labels,
    add_pca_inset,
    plot_spectrum
)

from obj_consts import get_obj_label

is_eval = True
# is_eval = False
is_gt = True

variant = 'base-full'
ckpt = 31
# variant = 'base4-full'
# ckpt = 34
variant = 'split_clamp-full'
ckpt = 33

def load_embeddings(variant, ckpt):
    pth = f"/srv/share/jye72/objectnav/{variant}/{variant}.{ckpt}.pth"
    weights = torch.load(pth, map_location='cpu')['state_dict']
    goal_embedding = weights['actor_critic.goal_embedder.weight']
    sem_embedding = weights["actor_critic.visual_encoders.encoders.['depth', 'rgb', 'semantic'].0.semantic_embedder.weight"]
    return goal_embedding, sem_embedding # [21x32], [42x4]

goal_embedding, sem_embedding = load_embeddings(variant, ckpt)

# TODO success plot
def get_successes(variant, ckpt):
    df, _ = load_stats(variant, ckpt)
    mean_success = dict(df.groupby('obj_cat')['success'].transform('mean')) # .apply(float).to_dict()
    print(mean_success)
print(get_successes(variant, ckpt))
# df = load_variant(variant, ckpt) # Just to check success

# * Goal embedding
from fp_finder import extract_flat_key
fig = plt.figure()
pca = PCA(n_components=21)
pca.fit(goal_embedding)

prep_plt()
plt.plot(np.cumsum(pca.explained_variance_ratio_))


#%%
variant = 'aux_all-curpol'
ckpt = 38
pth = f"/srv/share/jye72/objectnav/{variant}/{variant}.{ckpt}.pth"
weights = torch.load(pth, map_location='cpu')['state_dict']
for i in range(6):
    embedding = weights[f'aux_tasks.{i + 6}.decoder.weight']
    print(weights[f'aux_tasks.{i + 6}.decoder.bias'])
    plt.plot(embedding[0])


#%%
# mapped_ids = map_row_to_key(df, ids[:2000], key='obj_cat')
# unique, inverse = np.unique(mapped_ids, return_inverse=True)
# palette = sns.color_palette('husl', n_colors=len(unique))
fig = plt.figure()
view_indices = [0, 1]

if len(view_indices) == 3:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = plt.gca()

# embedding = sem_embedding
embedding = goal_embedding
# pca = PCA(n_components=4)
pca = PCA(n_components=21)
pca.fit(embedding)

reduced = pca.transform(embedding)[:, view_indices]
# eff_dim = part_ratio(embedding)

# outliers = np.array(['clothes', 'tv_monitor', 'fireplace', 'gym_equipment'])
# outliers = np.array(['gym_equipment'])
# indices = [goal_labels.index(o) for o in outliers]
# mask = np.ones(len(reduced), np.bool)
# mask[indices] = 0
# reduced = reduced[mask]

ax.scatter(*reduced.T) # Sparser plotting
ax.axis('off')

# Add text
for i, el in enumerate(reduced):
    obj_cat = goal_labels[i]
    ax.text(*el.T, get_obj_label(obj_cat), rotation=0, size=8)

ax.text(-0.5, 1.5, "Goal Embeddings", size=16)
# ax.set_title(f"Goal Embed, Part. Ratio: {eff_dim:.1g}")
# ax.set_title(f"Goal Embed, Part. Ratio: {eff_dim:.1g}")
add_pca_inset(fig, pca, loc=(0.8, 0.5, 0.2, 0.2))

fig.savefig('test.pdf', bbox_inches='tight')

#%%
# TODO get labels for these guys
view_indices = [0, 1]
# view_indices = [0, 1, 2]
pca = PCA(n_components=4)
pca.fit(sem_embedding)
print(pca.explained_variance_ratio_)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
reduced = pca.transform(sem_embedding)[:, view_indices]
fig = plt.figure(figsize=(24, 24))
if len(view_indices) == 3:
    ax = fig.add_subplot(111, projection='3d')
else:
    ax = plt.gca()
ax.scatter(*reduced.T)

for i, el in enumerate(reduced):
    obj_cat = mpcat40_labels[i]
    ax.text(*el.T, get_obj_label(obj_cat), rotation=0, size=18)


#%%
# Is it all normalized?
print(sem_embedding.norm(dim=-1))

#%%
import networkx as nx
def get_l2_distances(embeddings):
    # embeddings N x H
    # return N x N of norm
    N, H = embeddings.size()
    diff = embeddings.unsqueeze(1).expand(N, N, H) # N x N x H
    diff = diff - embeddings.unsqueeze(0)
    return torch.linalg.norm(diff, dim=-1)

plotted = (goal_embedding, goal_labels)
# plotted = (sem_embedding, mpcat40_labels)
dists = get_l2_distances(plotted[0])
# dists = get_l2_distances(goal_embedding)
plt.imshow(dists)
plt.colorbar()
plt.xticks(
    ticks=np.arange(len(plotted[0])),
    labels=map(get_obj_label, plotted[1]),
    rotation=90
)
plt.yticks(
    ticks=np.arange(len(plotted[0])),
    labels=map(get_obj_label, plotted[1]),
    rotation=0
)
print("done")

#%%
# from sklearn.manifold import MDS
# embedding = MDS(n_components=2, random_state=1)
# sem_transformed = embedding.fit_transform(sem_embedding)
# plt.scatter(*sem_transformed.T)
# for i, el in enumerate(sem_transformed):
#     obj_cat = mpcat40_labels[i]
#     plt.text(*el.T, get_obj_label(obj_cat), rotation=10)


#%%
def plot_spring_graph(dists, k=3):
    # Provide a dense edge matrix.
    G = nx.Graph()
    [G.add_node(obj) for obj in goal_labels]
    for i in range(dists.shape[0]):
        nns = np.argsort(dists[i])[:k]
        for j in nns:
        # for j in range(i, dists.shape[1]):
            G.add_edge(goal_labels[i], goal_labels[j], weight=1.0/(1.0 + dists[i, j]))
    # pos = nx.spring_layout(G,scale=4)
    # pos = nx.nx_agraph.graphviz_layout(Gs, prog='fdp')
    pos = nx.nx_agraph.graphviz_layout(G)

    weights = nx.get_edge_attributes(G,'weight').values()
    nx.draw(G, pos, font_size=8, with_labels=True, width=list(weights))
    plt.title(variant)
    plt.show()

plot_spring_graph(dists)