#%%
# This notebook analyzes misc episode-level statistics; i.e. reproduces Fig A.1.
import numpy as np
import pandas as pd
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
import torch
from obj_consts import get_variant_labels, get_obj_label
from fp_finder import load_variant
from analyze_utils import prep_plt, load_stats

variant = 'base-full'
ckpt = 31

# variant = 'base4-full'
# ckpt = 34
# ckpt = 33

variant = 'split_clamp-full'
ckpt = 31

variant = 'split_rednet-full'
ckpt = 38

is_eval = True
# is_eval = False
is_gt = False
# is_gt = True

# meta_df, title = load_stats(variant, ckpt, is_gt=is_gt)
meta_df, title = load_stats(variant, ckpt, override_fn=f'{variant}/{ckpt}/eval_gt_False_21.pth')
meta_df['variant'] = 'Tethered'

print("Success\t", meta_df['success'].mean())
print("Coverage\t", meta_df['coverage'].mean())
print("Coverage on Success\t", meta_df[meta_df['success'] == 1.0]['coverage'].mean())

variant = 'split_clamp-full'
ckpt = 31
meta_df_2, _ = load_stats(variant, ckpt, is_gt=is_gt)
meta_df_2['variant'] = 'Base'

meta_df = pd.concat([meta_df, meta_df_2])
# meta_df['obj_d2g'] = meta_df.groupby('obj_cat')['geodesic'].transform('mean')
# meta_df = meta_df.sort_values('obj_d2g', ascending=True)

meta_df['scene'] = pd.Categorical(meta_df.scene)
#%%
#%%
# * Success vs Obj Goal / Scene
prep_plt()
y = "success"
# y = "coverage"
# y = "steps"
# y = "spl"
palette = sns.color_palette(n_colors=2)

x = 'obj_cat'
# x = 'scene'
if x == 'obj_cat':
    meta_df = meta_df.sort_values('obj_freq', ascending=False)
    ax = sns.barplot(x=x, y=y, data=meta_df, ci=None, hue='variant', palette=palette)

if x == 'scene':
    meta_df['scene_diff'] = meta_df.groupby('scene')['success'].transform('mean')
    meta_df = meta_df.sort_values('scene_diff', ascending=False)
    # Hmm, sort doesn't seem to work by default
    scene_diff_order = meta_df['scene'].unique()
    # print(scene_diff_order)
    ax = sns.barplot(x=x, y=y, data=meta_df, ci=None, hue='variant', palette=palette, order=scene_diff_order)


# ax = sns.barplot(x="obj_cat", y="success", data=meta_df, ci=None)
# ax.set_xlabel("Goal Category in Ascending Average Distance")
# ax.set_xlabel("Goal Category in Descending Frequency")
# ax.set_ylim(0.0, 0.85)

sns.despine(ax=ax)
ax.set_ylabel(f"{y} Ratio")
ax.set_ylabel(f"Average Success")
if x == "obj_cat":
    ax.set_xlabel("Goals (Descending Frequency)")
elif x == 'scene':
    ax.set_xlabel("Scene")
ax.set_title("")
ax.legend(["Tethered", "Base"], frameon=False, fontsize=16)

# ax.text(8, 0.7, "Tethered", color=palette[0], size=16)
# ax.text(8, 0.64, "Base", color=palette[1], size=16)

strs = map(lambda label: label._text, ax.get_xticklabels())
if x == 'obj_cat':
    mapper = get_obj_label
elif x == 'scene':
    mapper = lambda x: x.split('/')[-2][:5]
ax.set_xticklabels(map(mapper, strs), rotation=45, horizontalalignment='right')

plt.savefig('test.pdf', dpi=150, bbox_inches="tight")
#%%

meta_df_dummy = meta_df_2.copy(deep=True)
meta_df_dummy['success'] = 1
meta_df_dummy['variant'] = 'Total Episodes'
df3 = pd.concat([meta_df_dummy, meta_df])
df3 = df3[df3['success'] == 1]
def plot_success_vs_geodesic(df, cat=None, scene=None, ax=None):
    plot_df = df
    if cat is not None:
        plot_df = df[df['obj_cat'] == cat]
    if scene is not None:
        plot_df = df[df['scene'] == scene]
    # prep_plt()
    # sns.despine(ax=ax)
    g = sns.displot(
        data=plot_df,
        x="geodesic",
        hue="variant",
        # hue="success",
        multiple="dodge",
        # col='variant',
        ax=ax,
        bins=np.arange(0, 30, 2)
    )
    g.set_axis_labels('Goal Geodesic Distance', 'Success Count')
    g.legend.set_title("")
    g.legend.set_bbox_to_anchor((0.7, 0.7))
    # ax.set_xlabel("Geodesic distance")
    # ax.set_title(f"Base")
    # ax.set_title(f"Tethered")
    # ax.set_title(f"{title}")

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plot_success_vs_geodesic(meta_df, ax=ax)
plot_success_vs_geodesic(df3)
plt.savefig('test.pdf', bbox_inches='tight')

# plot_success_vs_geodesic(meta_df, cat="chair")
# plot_success_vs_geodesic(meta_df, cat="table")
# plot_success_vs_geodesic(meta_df, cat="cushion")
# plot_success_vs_geodesic(meta_df, cat="cabinet")

#%%
# Other random plots below
success = 1.0
success = 0.0
success_df = meta_df[meta_df['success'] == success]
# plot_success_vs_geodesic(meta_df, ax=plt.gca())

ax = sns.histplot(
    data=success_df,
    x="geodesic",
    hue="cat_or_rare",
    multiple="stack",
    ax=plt.gca(),
    shrink=0.5,
    bins=np.arange(30),
)
import matplotlib as mpl
legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]
legends[0].set_title("Category")
plt.ylim(0, 300)
plt.xlabel("Geodesic Distance")
plt.ylabel(f"{'Failure' if success == 0.0 else 'Success'} Count")
# plt.ylabel("Success Count")

#%%
ax = sns.countplot(data=meta_df, x="obj_cat")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.set_title(f"GT Category Distribution {'EVAL' if is_eval else 'TRAIN'}")

#%%
# ax = sns.barplot(data=meta_df, x="success", y="geodesic", hue="obj_cat")
# ax = sns.barplot(data=meta_df, x="obj_cat", y="geodesic")
ax = sns.barplot(data=meta_df, x="obj_cat", y="geodesic", hue="success")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_xlabel("Category")
ax.set_ylabel("Geodesic Distance")
ax.set_title(f"{title} Distance per Category")