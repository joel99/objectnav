# General purpose helper functions

import os.path as osp
import seaborn as sns
import numpy as np
import numpy.linalg as linalg
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.utils.extmath import svd_flip
from sklearn.decomposition import PCA

from obj_consts import get_variant_labels

def prep_plt(ax=None):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    LARGE_SIZE = 15
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-muted')
    plt.figure(figsize=(6,4))
    if ax is None:
        ax = plt.gca()
    spine_alpha = 0.3
    ax.spines['right'].set_alpha(spine_alpha)
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    ax.spines['top'].set_alpha(spine_alpha)
    ax.grid(alpha=0.25)
    plt.tight_layout()

def get_belief(df, ep_id=0, index=0, fused=False): # returns T x H
    row = df.iloc[ep_id]
    el = row['fused_belief'] if fused else row['beliefs'][:, index]
    # prepend a zero
    el = torch.cat([
        torch.zeros(1, el.shape[1], dtype=el.dtype, device=el.device),
        el
    ], dim=0)
    return el.float().numpy()

def plot_to_axis(fig, ax, traj, palette=None, normalize=True, scatter=False, colorbar=False):
    if palette is None:
        palette = sns.color_palette("flare", as_cmap=True)
    values = []
    colors = []
    for i in range(traj.shape[0]-1):
        values.append(i/(traj.shape[0] if normalize else 500))
        # colors.append(palette(values[-1]))
        ax.plot(
            *traj[i:i+2].T,
            color=palette(values[-1])
        )
        if scatter:
            ax.scatter(
                *traj[i:i+1].T,
                color=palette(values[-1])
            )

    ax.axis('off')
    if colorbar: # Only supports 1 call.
        norm = mpl.colors.Normalize(vmin=0,vmax=len(values))
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        fig.colorbar(sm, cax=cax,
            # ticks=values[::4],
            ticks=np.linspace(0, len(values), 8),
            # ticks=np.arange(len(values))[::4],
            # boundaries=np.arange(len(values))[::4],
            orientation='horizontal'
        )

def svd(jac):
    U, S, Vt = linalg.svd(jac, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    return U, S, Vt

def load_device(try_load=0):
    # try_load -- use this if you have a GPU alloc-ed (in interactive mode)
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda", min(try_load, torch.cuda.device_count() - 1))
    else:
        device = torch.device("cpu")
    print(f'Loaded: {device}')
    return device

def part_ratio(fps, n=30, pca=None):
    if pca is None:
        pca = PCA(n_components=min(n, len(fps)))
        pca.fit(fps)
    sv = np.array(pca.singular_values_)
    return sum(sv) ** 2 / sum(sv ** 2)

def loc_part_ratio(points, sample_size=20, k=50, seed=0): # From Aitken 2020 A2
    # TODO, would be best to parallelize
    np.random.seed(seed)
    # points: N x H
    sampled = np.random.choice(len(points), sample_size)
    sampled = points[sampled] # S x H
    # get k nearest neighbors for each point
    loc_ratios = []

    for point in sampled:
        diffs = points - point # N x H
        distances = diffs.pow(2).sum(dim=1) # N
        knn = torch.argsort(distances)[:k]
        knn_points = points[knn] # K x H
        loc_ratios.append(part_ratio(knn_points))
        # Now do pca and argsort here
    return loc_ratios, sum(loc_ratios) / len(loc_ratios)

task_cat2mpcat40_labels = [
    'chair',
    'table',
    'picture',
    'cabinet',
    'cushion',
    'sofa',
    'bed',
    'chest_of_drawers',
    'plant',
    'sink',
    'toilet',
    'stool',
    'towel',
    'tv_monitor',
    'shower',
    'bathtub',
    'counter',
    'fireplace',
    'gym_equipment',
    'seating',
    'clothes',
]

# Pulled from https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv
mpcat40_labels = [
    # '', # -1
    'void',
    'wall',
    'floor',
    'chair',
    'door',
    'table', # 5
    'picture',
    'cabinet',
    'cushion',
    'window',
    'sofa', # 10
    'bed',
    'curtain',
    'chest_of_drawers',
    'plant',
    'sink',
    'stairs',
    'ceiling',
    'toilet',
    'stool',
    'towel', # 20
    'mirror',
    'tv_monitor',
    'shower',
    'column',
    'bathtub',
    'counter',
    'fireplace',
    'lighting',
    'beam',
    'railing',
    'shelving',
    'blinds',
    'gym_equipment', # 33
    'seating',
    'board_panel',
    'furniture',
    'appliances',
    'clothes',
    'objects',
    'misc',
    'unlabeled' # 41
]

# [
#     3,  # ('chair', 2, 0)
#     5,  # ('table', 4, 1)
#     6,  # ('picture', 5, 2)
#     7,  # ('cabinet', 6, 3)
#     8,  # ('cushion', 7, 4)
#     10,  # ('sofa', 9, 5),
#     11,  # ('bed', 10, 6)
#     13,  # ('chest_of_drawers', 12, 7),
#     14,  # ('plant', 13, 8)
#     15,  # ('sink', 14, 9)
#     18,  # ('toilet', 17, 10),
#     19,  # ('stool', 18, 11),
#     20,  # ('towel', 19, 12)
#     22,  # ('tv_monitor', 21, 13)
#     23,  # ('shower', 22, 14)
#     25,  # ('bathtub', 24, 15)
#     26,  # ('counter', 25, 16),
#     27,  # ('fireplace', 26, 17),
#     33,  # ('gym_equipment', 32, 18),
#     34,  # ('seating', 33, 19),
#     38,  # ('clothes', 37, 20),
#     43,  # ('foodstuff', 42, 21),
#     44,  # ('stationery', 43, 22),
#     45,  # ('fruit', 44, 23),
#     46,  # ('plaything', 45, 24),
#     47,  # ('hand_tool', 46, 25),
#     48,  # ('game_equipment', 47, 26),
#     49,  # ('kitchenware', 48, 27)
# ]

def get_variant_path(
    variant, ckpt, is_eval=True, is_gt=True, eval_stat_root='/nethome/jye72/share/objectnav_eval/',
    override_fn=None
):
    if override_fn:
        fn = override_fn
    else:
        if is_eval:
            fn = f"{variant}/{ckpt}/{'eval' if is_eval else 'train'}_gt_{str(is_gt)}.pth"
        else:
            fn = f"{variant}/{ckpt}/train.pth"
            if not osp.exists(osp.join(eval_stat_root, fn)):
                fn = f"{variant}/{ckpt}/train_gt_{str(is_gt)}.pth"
    return osp.join(eval_stat_root, fn)

def load_stats(
    variant, ckpt, is_eval=True, is_gt=True,
    eval_stat_root = '/nethome/jye72/share/objectnav_eval/',
    override_fn = None,
):
    eval_fn = get_variant_path(variant, ckpt, is_eval=is_eval, is_gt=is_gt, override_fn=override_fn)
    data = torch.load(eval_fn)
    num_updates = data['step_id']
    data = data['payload']

    meta_df = []
    for i, ep in enumerate(data):
        stats = ep['stats']
        did_stop = ep['did_stop']
        info = ep['info']
        ep_info = info['episode_info']
        meta_df.append({
            "ep": ep_info['episode_id'],
            "scene": ep_info['scene_id'],
            "geodesic": ep_info['info']['geodesic_distance'],
            "success": stats['success'],
            "spl": stats['spl'],
            # "did_stop": did_stop,
            "obj_cat": ep_info['object_category'],
            "d2g": stats['distance_to_goal'],
            "coverage": stats['coverage.reached'],
            "visit_count": stats['coverage.visit_count'],
            "steps": stats['coverage.step'],
            # "sge": float(stats['goal_vis']),
        })
    meta_df = pd.DataFrame(meta_df) #, index=['ep', 'scene'], columns=['success'])
    meta_df = meta_df.sort_values('obj_cat', ascending=True)
    meta_df['obj_freq'] = meta_df.groupby('obj_cat')['obj_cat'].transform('count')
    meta_df = meta_df.sort_values('obj_freq', ascending=False)
    label = get_variant_labels(variant, is_gt=is_gt)
    title = f"{label} ~ {num_updates:.3g}M Frames"
    return meta_df, title

# From https://github.com/audeering/audtorch/blob/master/audtorch/metrics/functional.py
def pearsonr(
        x,
        y,
        batch_first=True,
):
    """
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
    if len(x.shape) == 0:
        print(x)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr

