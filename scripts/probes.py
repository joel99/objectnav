#%%
# Notebook for probe expeirments
import numpy as np
import numpy.linalg as linalg
import torch
import pandas as pd
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA

import seaborn as sns
import PIL.Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import jacobian

from sklearn.metrics import r2_score, explained_variance_score
from scipy.stats import pearsonr

from analyze_utils import (
    prep_plt, plot_to_axis, get_belief, load_device,
    task_cat2mpcat40_labels
)

# from habitat.tasks.nav.object_nav_task import REGION_ANNOTATIONS
REGION_ANNOTATIONS = {
    'bathroom': 0,
    'bedroom': 1,
    'closet': 2,
    'dining room': 3,
    'entryway/foyer/lobby': 4, # (should be the front door, not any door)
    'familyroom/lounge': 5, # (should be a room that a family hangs out in, not any area with couches)
    'garage': 6,
    'hallway': 7,
    'library': 8, # (should be room like a library at a university, not an individual study)
    'laundryroom/mudroom': 9, # (place where people do laundry, etc.)
    'kitchen': 10,
    'living room': 11, # (should be the main “showcase” living room in a house, not any area with couches)
    'meetingroom/conferenceroom': 12,
    'lounge': 13, # (any area where people relax in comfy chairs/couches that is not the family room or living room
    'office': 14, # (usually for an individual, or a small set of people)
    'porch/terrace/deck': 15, # (must be outdoors on ground level)
    'rec/game': 16, # (should have recreational objects, like pool table, etc.)
    'stairs': 17,
    'toilet': 18, # (should be a small room with ONLY a toilet)
    'utilityroom/toolroom': 19,
    'tv': 20, # (must have theater-style seating)
    'workout/gym/exercise': 21,
    'outdoor': 22, # areas containing grass, plants, bushes, trees, etc.
    'balcony': 23, # (must be outside and must not be on ground floor)
    'other room': 24, # (it is clearly a room, but the function is not clear)
    'bar': 25,
    'classroom': 26,
    'dining booth': 27,
    'spa/sauna': 28,
    'junk': 29, # (reflections of mirrors, random points floating in space, etc.)
    'no label': 30
}
REGION_ANNOT_LIST = {v: k for k, v in REGION_ANNOTATIONS.items()}
REGION_ANNOT_LIST = [REGION_ANNOT_LIST[i] for i in range(len(REGION_ANNOT_LIST))]
from fp_finder import (
    load_variant, load_recurrent_model,
    get_model_inputs, noise_ics, run_fp_finder, get_cache_path,
    get_jac_pt,
    extract_flat_key
)

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split


variants = ['base4-full', 'base-full', 'split_clamp-full']
ckpts = [35, 31, 31] # base4 34 has anomalously bad val. swapped to 35
variants = ['base-full', 'split_clamp-full']
ckpts = [31, 31]
dfs = {}

for v, c in zip(variants, ckpts):
    dfs[v] = {
        'train': load_variant(v, c, is_eval=False),
        'eval': load_variant(v, c, is_eval=True)
    }

#%%
# load_variant('base4-full', 34, is_eval=False, is_gt=True)
# load_variant('base4-full', 33, is_eval=False, is_gt=False)
# load_variant('base-full', 31, is_eval=False, is_gt=True)
# load_variant('base-full', 31, is_eval=False, is_gt=False)
df = load_variant('base-full', 31, is_eval=True, is_gt=True)
# load_variant('split_clamp-full', 31, is_eval=False, is_gt=True)
# load_variant('split_clamp-full', 31, is_eval=False, is_gt=False)
# df = load_variant('split_clamp-full', 31, is_eval=True, is_gt=False)

df.groupby('scene')['success'].agg('mean', np.std)
#%%
# * FYI pred seg doesn't render SGE. This is all GT
def inject_max(df):
    df['sge_max'] = df.apply(lambda x: np.maximum.accumulate(x.sge_t), axis=1)

def sge_first_index(sge_max):
    argwhere = np.argwhere(sge_max)[0]
    if argwhere.size(0) == 0:
        return -1
    return argwhere[0].item()

def inject_goal_seen_filter(df):
    df['sge_first_index'] = df.apply(lambda x: sge_first_index(x['sge_max'] > 0.05), axis=1)
    # df['sge_first_index'] = df.apply(lambda x: np.argwhere(x['sge_max'] > 0.05)[0][0].item(), axis=1)

def create_subset_trajs(df):
    for key in ['beliefs', 'fused_belief', 'fused_obs', 'd2g_t', 'timesteps']:
        df[f'subset_{key}'] = df.apply(lambda x: x[key][x['sge_first_index']:], axis=1)

for v in dfs:
    inject_max(dfs[v]['train'])
    inject_max(dfs[v]['eval'])
    inject_goal_seen_filter(dfs[v]['train'])
    inject_goal_seen_filter(dfs[v]['eval'])
    create_subset_trajs(dfs[v]['train'])
    create_subset_trajs(dfs[v]['eval'])

# plt.plot(dfs['base-full']['train'].iloc[1]['sge_max'])
# plt.plot(dfs['base4-full']['train'].iloc[0]['sge_max'])
# plt.plot(dfs['base4-full']['eval'].iloc[0]['sge_max'])
# plt.plot(dfs['base-full']['eval'].iloc[0]['sge_max'])

# print(dfs['base-full']['train'].iloc[1]['sge_first_index'])
# print(dfs['base-full']['train'].iloc[1]['subset_fused_belief'].size())
# print(dfs['base-full']['train'].iloc[1]['fused_belief'].size())
# print(dfs['base-full']['train'].iloc[1]['subset_beliefs'].size())

#%%
x_train, _ = extract_flat_key(dfs['split_clamp-full']['train'], 'fused_belief')
x_train, _ = extract_flat_key(dfs['split_clamp-full']['eval'], 'fused_belief')
print(x_train.size())

#%%
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import cross_val_score
from math import log10, floor
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)
def score_str(scores):
    return [round_sig(s) for s in scores]

def probe(df_train, df_eval, key1, key2, index=-1, alpha=1e5, transform=None, plot=False, regression=True):
    # ~ 100K steps
    x_train, _ = extract_flat_key(df_train, key1, index=index)
    x_test, _ = extract_flat_key(df_eval, key1, index=index)
    if len(x_train.shape) == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    y_train, _ = extract_flat_key(df_train, key2)
    y_test, _ = extract_flat_key(df_eval, key2)
    if transform is not None:
        x_train, y_train = transform(x_train, y_train)
        x_test, y_test = transform(x_test, y_test)

    if regression:
        # clf = RidgeCV(alpha=alpha)
        clf = Ridge(alpha=alpha) # We don't use CV to maintain consistency across probes + beliefs
    else:
        clf = LogisticRegression(C=alpha) # Test this out
    # scores = cross_val_score(clf, x_train, y_train, cv=5)
    # print(f"CV: {score_str(scores)}")

    # x_tr, x_val, y_tr, y_val = train_test_split(
    #     x_train, y_train, test_size=0.2, random_state=42)
    # clf.fit(x_tr, y_tr)
    clf.fit(x_train, y_train)
    key_annot = f"{key1}-{index}" if index >= 0 else key1
    print(f'{key_annot} train score: {clf.score(x_train, y_train):.2g}')
    y_pred = clf.predict(x_test)
    if plot:
        prep_plt()
        plt.scatter(y_pred[::50], y_test[::50])
        plt.xlabel(f'{key2} pred')
        plt.ylabel(f'{key2} test')
        plt.title(f'{key_annot}-{key2} Score: {clf.score(x_test, y_test):.2g}')
    return clf, clf.score(x_test, y_test)

text_pairs = ['C4', 'PBL', 'C16', 'GID', 'CP', 'ADP', 'Fused', 'Obs', 'Steps']
def probe_all(dfs, target, mute=False, sge_subset=False, use_steps=False, **kwargs):
    # Probe all sources
    all_scores = []
    keys = ['beliefs', 'fused_belief', 'fused_obs', 'timesteps']
    if sge_subset:
        keys =[f'subset_{k}' for k in keys]
    for i in range(6):
        all_scores.append(
            probe(dfs['train'], dfs['eval'], keys[0], target, index=i, **kwargs)[1]
        )
    all_scores.append(
        probe(dfs['train'], dfs['eval'], keys[1], target, **kwargs)[1]
    )
    all_scores.append(
        probe(dfs['train'], dfs['eval'], keys[2], target, **kwargs)[1]
    )
    if use_steps:
        all_scores.append(
            probe(dfs['train'], dfs['eval'], keys[3], target, **kwargs)[1]
        )
    if not mute:
        print(f'{target} Probes (a={alpha:.1g})')
        for text, s in zip(text_pairs, all_scores):
            print(f"{text}: {round_sig(s)}")
    return all_scores

def probe_series(dfs, target, transforms, **kwargs):
    # transforms: label to lambda
    # Bar plot
    df_info = []
    for k, v in transforms.items():
        var_scores = []
        for var in variants:
            var_scores.append(probe_all(dfs[var], target, mute=True, transform=v, **kwargs))
        for i, text in enumerate(text_pairs):
            for var, var_score in zip(variants, var_scores):
                info = {'source': text, 'transform': k, 'score': var_score[i], 'variant': var}
                if text in ['Fused', 'Obs', 'Steps']:
                    info['cat'] = text
                else:
                    info['cat'] = 'Belief'
                df_info.append(info)
    df = pd.DataFrame(df_info)
    # prep_plt()
    # sns.barplot(x='source', y='score', hue='transform', data=df)
    # plt.legend()
    # plt.title(f'{variant}-{ckpt}, {target}')
    return df

all_transforms = {
    'none': None,
    # 'crop100': lambda x, y: (x, torch.minimum(y, torch.tensor(100))),
    # 'crop350': lambda x, y: (x, torch.clamp(y, max=350)),
    # 'sqrt': lambda x, y: (x, torch.sqrt(y)),
    # 'log': lambda x, y: (x, torch.log(y + 1)),
    # 'log_sge': lambda x, y: (x, torch.log(y + 0.001)), # Nope
    # 'crop10': lambda x, y: (x, torch.clamp(y, max=0.1)),
    # 'thresh': lambda x, t: (x, (y > 0.05).to(dtype=torch.int)),
    # 'filter': lambda x, y: (x[y != REGION_ANNOTATIONS['no label']], y[y != REGION_ANNOTATIONS['no label']])
}

target = 'timesteps'
# CP gets above the rest (transform = log does better than none but doesn't change trends)
# Unfortunately, it's not clearly above the others so I dunno what to say here.
# All beliefs to appear to be tracking it weakly though.


target = 'visit_count_t'
target = 'd2g_t' # Quite weak. < 0.2.
# target = 'subset_d2g_t' # Terrible. 6-Action is truly terrible.
# target = 'sge_max'


# target = 'room_cat'
df = probe_series(dfs, target, transforms=all_transforms, plot=False,
    sge_subset='subset' in target,
    regression=target is not 'room_cat',
    # alpha=0.1,
    use_steps=True
)

# target = 'sge_max'
# df_sge_max = probe_series(dfs, target, transforms=all_transforms, plot=False,
#     sge_subset='subset' in target,
#     regression=target is not 'room_cat',
#     # alpha=0.1,
#     use_steps=True
# )



#%%
# * Probe primary
palette = sns.color_palette(n_colors=2)
palette = sns.color_palette(n_colors=3)
df['Probe'] = 'Time'
df_sge_max['Probe'] = 'Max SGE'
df_merged = pd.concat([df, df_sge_max])
df_merged = df_merged[df_merged['source'] != 'Steps']

# ok the game plan is to merge the beliefs (show high variance)
g = sns.catplot(x="cat", y="score",
    hue="variant", col="Probe",
# g = sns.catplot(x="variant", y="score",
#     hue="cat", col="Probe",
    data=df_merged, kind="bar",
    height=4, aspect=.7, palette=palette)

(g.set_axis_labels("Variant", "Probe $R^2$")
# (g.set_axis_labels("Source", "Probe $R^2$")
    # .set_xticklabels(['Beliefs', 'Fused', 'Obs'])
    .set_xticklabels(['6-Act', '6-Act + Tether'])
    .set_titles("")
)

g.axes[0][0].text(-.35, 0.5, "Time", size=22)
g.axes[0][1].text(-.35, 0.45, "Max\nSGE", size=22)

# g.axes[0][0].text(1., 0.5, "Time", size=22)
# g.axes[0][1].text(1., 0.5, "Max SGE", size=22)

# g.axes[0][1].text(1.7, 0.03, "6-Act", size=16, color=palette[0], rotation=90)
# g.axes[0][1].text(2.1, 0.03, "6-Act + Tether", size=16, color=palette[1], rotation=90)

g.axes[0][0].text(.65, 0.03, "Beliefs", size=16, color='white', rotation=90)
g.axes[0][0].text(.92, 0.03, "Fused", size=16, color='white', rotation=90)
g.axes[0][0].text(1.19, 0.03, "Obs", size=16, color=palette[2], rotation=90)

g.axes[0][1].text(.65, 0.03, "Beliefs", size=16, color='white', rotation=90)
g.axes[0][1].text(.92, 0.03, "Fused", size=16, color='white', rotation=90)
g.axes[0][1].text(1.19, 0.03, "Obs", size=16, color=palette[2], rotation=90)


# g.axes[0][1].text(1.1, 0.4, "6-Act", size=16, color='blue')
# g.axes[0][1].text(.7, 0.35, "6-Act + Tether", size=16, color='green')

g.legend.remove()
# g.add_legend(bbox_to_anchor=(0.82, 0.65), frameon=False, title='Variant')
# legend = g.legend
# new_labels = ['6-Act', '6-Act + Tether']
# for t, l in zip(legend.texts, new_labels): t.set_text(l)
# legend.set_title('Variant')
# legend.set_bbox_to_anchor((0.55, 0.9))
# legend.get_frame().set_visible(True)
# legend.get_frame().set_linewidth(1)
# legend.get_frame().set_edgecolor('b')
plt.savefig('test.pdf', bbox_inches='tight')
#%%
# * Probe Supp
df1 = probe_series(dfs, 'visit_count_t', transforms=all_transforms, plot=False,
    sge_subset='subset' in target,
    regression=target is not 'room_cat',
    # alpha=0.1,
    use_steps=True
)
df1['Probe'] = 'Time'

df2 = probe_series(dfs, 'd2g_t', transforms=all_transforms, plot=False,
    sge_subset='subset' in target,
    regression=target is not 'room_cat',
    # alpha=0.1,
    use_steps=True
)
df2['Probe'] = 'Dist To Goal'

df3 = probe_series(dfs, 'room_cat', transforms=all_transforms, plot=False,
    sge_subset='subset' in target,
    regression=False,
    alpha=0.1,
    use_steps=True
)
df3['Probe'] = 'Room Cat.'

#%%

df = pd.concat([df1, df2, df3])

palette = sns.color_palette(n_colors=2)
palette = sns.color_palette(n_colors=3)

# ok the game plan is to merge the beliefs (show high variance)
g = sns.catplot(
    hue="variant",
    # x="cat", y="score",
    x="score", y="cat",
    row="Probe",orient='h',
    data=df, kind="bar",
# g = sns.catplot(x="variant", y="score",
#     hue="cat", col="Probe",
    height=4, aspect=.7, palette=palette,
    facet_kws={'height': 1, 'aspect': 5}
)
g.fig.set_figheight(5)
g.fig.set_figwidth(6)

# (g.set_axis_labels("Variant", "Probe $R^2$")
# (g.set_axis_labels("Source", "Probe Score")
#     .set_xticklabels(['Beliefs', 'Fused', 'Obs', 'Steps'])
#     .set_titles("")
# )

(g.set_axis_labels("Probe Score", "Source")
    .set_yticklabels(['Beliefs', 'Fused', 'Obs', 'Steps'])
    .set_titles("")
)


g.axes[0][0].text(0.6, 3, "Visit Count", size=18, rotation=-70, ha='center')
g.axes[1][0].text(0.55, 3, "Distance\nto Goal", size=18, rotation=-70, ha='center')
# g.axes[1][0].text(0.6, 3, "Distance\nto Goal", size=18, rotation=-70, ha='center')
g.axes[2][0].text(0.6, 3, "Room Cat.", size=18, rotation=-70, ha='center')
# g.axes[0][1].text(-.35, 0.45, "Distance\nto Goal", size=20)
# g.axes[0][2].text(-.35, 0.6, "Room Category", size=20)

g.axes[0][0].text(0.02, 2.95, "Base", size=11, color='white')
g.axes[0][0].text(0.02, 3.35, "Tether", size=11, color='white')
g.axes[1][0].text(0.02, 2.95, "Base", size=11, color='black')
g.axes[1][0].text(0.04, 3.35, "Tether", size=11, color='black')
g.axes[2][0].text(0.02, 2.95, "Base", size=11, color='white')
g.axes[2][0].text(0.02, 3.35, "Tether", size=11, color='white')
g.legend.remove()

g.savefig('test.pdf', bbox_inches='tight')

#%%

# * This cell and below are miscellaneous experiments. (Not included in paper)
prep_plt()

sub_df = df
if target == 'timesteps':
    sub_df = df[df['source'] != 'Steps']
# sns.barplot(x='source', y='base4-full_score', hue='transform', data=df)
# g = sns.barplot(x='source', y='score', hue='variant', data=sub_df)

# Collapse
g = sns.barplot(x='cat', y='score', hue='variant', data=sub_df)

# sns.barplot(x='source', y='base-full_score', hue='transform', data=df)
sns.despine(ax=g)
plt.legend(markerfirst=False)

legend = [c for c in g.get_children() if isinstance(c, mpl.legend.Legend)][0]
# title
# new_title = 'My title'
# legend.set_title(new_title)

new_labels = ['4-Act', '6-Act', '6-Act + Tether']
new_labels = ['6-Act', '6-Act + Tether']
for t, l in zip(legend.texts, new_labels): t.set_text(l)

plt.ylabel("Timestep $R^2$")
# plt.ylabel("Visit Count $R^2$")
# plt.ylabel("Distance To Goal $R^2$")
# plt.ylabel("Max SGE $R^2$")

# plt.ylabel("Room Category Accuracy")

# plt.ylabel(f"{target} Score")
plt.xlabel("Probed Representation")
    # dfs['base-full']['eval'],
# plt.title(f'{variant}-{ckpt}, {target}')

plt.savefig('test.pdf', dpi=150, bbox_inches="tight")



#%%
# print(torch.tensor(int(df_eval.iloc[0]['steps'] + 1)))
# df_eval.iloc[0]['steps']
# print(df_eval.iloc[0]['room_cat'].size())
#%%
# SEMANTICS

# from sklearn import svm
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y)


# TODO do some room cat filter out the useless guys
# TODO logistic regression
# Just correlate time spent in room with goal first
target = 'room_cat'
room_cat, _ = extract_flat_key(df_eval, 'room_cat')
room_cat = room_cat.to(torch.int)
goal, _ = extract_flat_key(df_eval, 'obj_cat_num')
# probe_all(target, plot=False, transform=all_transforms['none'])
filt_rooms = room_cat[room_cat != 30] # 30 = no_label
goal = goal[room_cat != 30]

filt_rooms = filt_rooms[::10]
goal = goal[::10]
df_info = []
for i, g in enumerate(goal):
    df_info.append({'goal': int(g.item()), 'room': filt_rooms[i].item()})
df = pd.DataFrame(df_info)
nonempty_rooms = []
default_to_nonempty_map = {}
for i, room in enumerate(REGION_ANNOT_LIST):
    if len(df[df['room'] == i]) > 0:
        default_to_nonempty_map[i] = len(nonempty_rooms)
        nonempty_rooms.append(room)
def get_nonempty(t):
    return default_to_nonempty_map[t.item()]
df['nonempty'] = df.apply(lambda x: get_nonempty(x.room), axis=1)
# df['goal_freq'] = df.groupby('goal')['goal'].transform('count')

# Construct a new df that is normalized
df_info = []
for i, g in enumerate(task_cat2mpcat40_labels):
    all_room_visits = df[df['goal'] == i]
    for j, room in enumerate(nonempty_rooms):
        room_visits = all_room_visits[all_room_visits['nonempty'] == j]
        # room_visits = df[(df['goal'] == goal) & (df['nonempty'] == j)]
        norm_visit = len(room_visits) / len(all_room_visits)
        df_info.append({'goal': i, 'room': j, 'rel_visit': norm_visit})
rel_df = pd.DataFrame(df_info)

#%%
# Correlation

f = plt.figure(figsize=(8, 6))
ax = sns.histplot(
    df, x="nonempty", y="goal",
    # df, x="room", y="goal",
    discrete=(True, True),
    stat='density',
    common_norm=False,
    cbar=True,
    # bins=30, discrete=(True, False), log_scale=(False, True),
)
ax.set_yticks(np.arange(len(task_cat2mpcat40_labels)))
ax.set_yticklabels(task_cat2mpcat40_labels)
ax.set_xticks(np.arange(len(nonempty_rooms)))
# ax.set_xticks(np.arange(len(REGION_ANNOT_LIST)))
ax.set_xticklabels(nonempty_rooms, rotation=30)
# ax.set_xticklabels(REGION_ANNOT_LIST, rotation=45)

#%%
# Normalized correlation
# We should really account for the number of unrelated rooms
# Or we're going to misinterpret goal trajs mostly in unlabeled rooms as too heavily in particular room
ax = sns.heatmap(rel_df.pivot('goal', 'room', 'rel_visit'))
ax.set_yticks(np.arange(len(task_cat2mpcat40_labels)))
ax.set_yticklabels(task_cat2mpcat40_labels, rotation=0)
# ax.set_xticks(np.arange(len(nonempty_rooms)))
ax.set_xticklabels(nonempty_rooms, rotation=90)


#%%
# Record weights
v = 'base4-full'
v = 'base-full' # This may be weaker but it's the one we want to fix.
clf, score = probe(
    dfs[v]['train'],
    dfs[v]['eval'],
    'beliefs', 'timesteps', index=1, plot=True
    # 'beliefs', 'timesteps', index=5, plot=True
    # 'beliefs', 'timesteps', index=4, plot=True
    # 'beliefs', 'timesteps', index=4, plot=True
    # 'beliefs', 'sge_max', index=1, plot=True
)

# Something is quite different about my predictions. Is there a bias term?
#%%
all_coefs = []

for i in range(6):
    clf = probe(dfs[v]['train'], dfs[v]['eval'], 'beliefs', 'timesteps', index=i)[0]
    all_coefs.append(np.concatenate([clf.coef_, clf.intercept_[None]], axis=0))
weights = torch.tensor(np.stack(all_coefs, axis=0)).to(torch.float)
print(weights.size())
torch.save(weights, f'/srv/flash1/jye72/share/{v}_timesteps.pth')

#%%
# Replace classifier weights with random vectors of the same magnitude.

def random_sub(weights, seed=0):
    torch.random.manual_seed(seed)
    axes = weights[:, :-1] # k x h
    random_directions = torch.rand(*axes.size())
    norm_directions = random_directions / random_directions.norm(dim=1, keepdim=True) # k x h
    return torch.cat([norm_directions * axes.norm(dim=1, keepdim=True), weights[:, -1:]], dim=-1)

for i in range(5):
    r = random_sub(weights, seed=i)
    torch.save(r, f'/srv/flash1/jye72/share/{v}_random_{i}_timesteps.pth')

#%%
# Any obvious cells?
print(weights.size())
for i in range(6):
    plt.plot(weights[i], label=i)
plt.legend()

#%%
# https://math.stackexchange.com/questions/453005/orthogonal-projection-onto-an-affine-subspace

# Let's test out this weight matrix
b = dfs[v]['eval'].iloc[0]['beliefs']
print(b.size())
clf = probe(dfs[v]['train'], dfs[v]['eval'], 'beliefs', 'timesteps', index=1)[0]
plt.plot(clf.predict(b[:,1]))
# print(clf.predict(b[:,0]))
# print(clf.predict(b[0,0].reshape(1, -1)))
# print(torch.dot(b[0, 0], torch.tensor(clf.coef_).to(torch.float)) + clf.intercept_)
# print(torch.dot(b[0, 0], weights[0][:-1]) + weights[0][-1])