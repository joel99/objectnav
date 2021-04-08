#%%
# Notebook for failure mode plots

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

from analyze_utils import (
    prep_plt, plot_to_axis, get_belief,
    load_device,
    loc_part_ratio, part_ratio, svd
)

# These csv annotations were logged in an external spreadsheet
df = pd.read_csv('qual_info.csv')

df.loc[df['first'] == 'debris', 'first'] = 'misc'
df.loc[df['first'] == 'terminal', 'first'] = 'misc'

df['agent'] = 'base'
df_base = df
print(df_base['first'].unique())

df = pd.read_csv('qual_tether.csv')
df['first'] = df['mode']
df['agent'] = 'tether'

df.loc[df['first'] == 'timeout', 'first'] = 'misc'
df.loc[df['first'] == 'portal', 'first'] = 'misc'
df.loc[df['first'] == 'visitation', 'first'] = 'loop'
print(df['first'].unique())

df = pd.concat([df, df_base])
df['dummy'] = 1 # for plotting

palette = sns.color_palette(palette='muted', n_colors=len(df['first'].unique()), desat=0.9)
palette_dict = {}
for i, mode in enumerate(df['first'].unique()):
    palette_dict[mode] = palette[i]

# To use in supplement
# Void failure modes are suspect. It's hard to tell whether it's causal. Worth mentioning it, likely.
use_detailed = True
use_detailed = False
if not use_detailed:
    df.loc[df['first'] == 'void', 'first'] = 'misc'
    df.loc[df['first'] == 'goal bug', 'first'] = 'misc'

#%%
def get_label(s):
    return s[0].upper() + s[1:]

# order = ['explore', 'plateau', 'loop', 'detection', 'last mile', 'commitment', 'void', 'open', 'misc', 'terminal', 'goal bug']
if use_detailed:
    order_base = ['explore', 'plateau', 'loop', 'detection', 'last mile', 'commitment', 'void', 'open', 'misc', 'goal bug']
else:
    order_base = ['explore', 'plateau', 'loop', 'misc', 'detection', 'last mile', 'commitment', 'open']
palette_base = [palette_dict[m] for m in order_base]
prep_plt()
subset_len = len(df[df['agent'] == 'base'])
ax = sns.barplot(x="first", y="dummy", orient="v",
    estimator=lambda x: len(x) / subset_len * 100,
    data=df[df['agent'] == 'base'], order=order_base, palette=palette_base, ci=None)

sns.despine(ax=ax)
strs = map(lambda label: label._text, ax.get_xticklabels())
for i, label in enumerate(strs):
    if label == 'commitment':
        label = 'commit'
    ax.text(i+0.02, 0.78, label, rotation=90, color='white', fontsize=16)
# ax.set_xticklabels(map(get_label, strs), rotation=40)
ax.set_xticklabels([])
ax.set_xticks([])

# Let's give this a shot
ax.text(3, 20, 'Failure Modes', fontsize=24)
ax.set_ylabel('Percent of Failures')
ax.set_xlabel('')
# ax.set_xlabel('Failure Mode')
plt.savefig('test.pdf', bbox_inches='tight')

#%%

if use_detailed:
    order_tether = ['quit', 'explore', 'plateau', 'loop', 'detection', 'void', 'last mile', 'misc', 'open', 'goal bug']
else:
    order_tether = ['quit', 'explore', 'plateau', 'loop', 'misc', 'detection', 'last mile', 'open']
palette_tether = [palette_dict[m] for m in order_tether]

prep_plt()

subset_df = df[df['agent'] == 'tether']
# https://github.com/mwaskom/seaborn/issues/1027
ax = sns.barplot(x="first", y="dummy", orient="v",
    estimator=lambda x: (len(x) / len(subset_df) * 100), # [df['agent'] == x['agent']]) * 100,
    data=subset_df, order=order_tether, palette=palette_tether, ci=None)

# ax = sns.barplot(x="first", y="mode_pct", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent of Failures")
sns.despine(ax=ax)

strs = map(lambda label: label._text, ax.get_xticklabels())
ax.set_xticklabels(map(get_label, strs), rotation=40)

# ax = sns.countplot(x='first', data=df, order=order)

ax.set_xlabel('Failure Mode')
plt.savefig('test.pdf', bbox_inches='tight')

# Pi plots are just hard to read
# data = df.groupby("mode")['mode'].count() #.sum()
# print(data)
# pie, ax = plt.subplots(figsize=[10,6])
# labels = data.keys()
# plt.pie(x=data, autopct="%.1f%%", explode=[0.05]*len(order), labels=order, pctdistance=0.5)
# data.plot.pie(autopct="%.1f%%")
