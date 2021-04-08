# Some basic code for fixed point finding
# Referencing Matt Golub's FP Finder.
# https://github.com/mattgolub/fixed-point-finder/blob/master/FixedPointFinder.py
# Runnable as a static script

"""
Notes on Diffs:
Matt's code is for TF and Python 2, and is class-oriented. We're not at that scale yet.
Matt has a different set of hyperparameters (seemingly way too strict)
Matt supports:
- Distance outlier detection + elimination TODO
- LR decay TODO
- Gradient norm TODO
- Cost Outlier detection + rerunning
- Some support for jacobian comps, logging, other RNN types.
"""

import argparse

import numpy as np
import numpy.linalg as linalg
import torch
import pandas as pd
import os
import os.path as osp
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import seaborn as sns
import PIL.Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import jacobian


from analyze_utils import (
    get_variant_path,
    prep_plt, plot_to_axis, get_belief, load_device,
    task_cat2mpcat40_labels
)
from fpf_utils import (
    AdaptiveLearningRate, AdaptiveGradNormClip
)

SAVE_ROOT = f'/srv/share/jye72/fps/'

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant", "-v", type=str, default="split-curric"
    )

    parser.add_argument(
        "--ckpt", "-c", type=int, default=35
    )

    parser.add_argument(
        "--index", "-i", type=int, default=0
    )

    parser.add_argument(
        "--scene", "-sc", type=str, default="all"
    )

    parser.add_argument(
        "--num-fp", "-n", type=int, default=10000
    )

    parser.add_argument(
        "--tag", "-t", type=str, default=''
    )

    parser.add_argument("--zero-sge", "-z", dest='zero_sge', action='store_true')
    parser.add_argument("--no-zero-sge", "-nz", dest='zero_sge', action='store_false')
    parser.set_defaults(zero_sge=False)

    parser.add_argument('--train', '-tr', dest='is_eval', action='store_false')
    parser.add_argument('--no-train', '-ntr', dest='is_eval', action='store_true')
    parser.set_defaults(is_eval=True)


    parser.add_argument('--override', '-o', dest='override', action='store_true')
    parser.add_argument('--no-override', '-no', dest='override', action='store_false')
    parser.set_defaults(override=False)

    parser.add_argument('--exclude', '-x', dest='exclude_outliers', action='store_true')
    parser.add_argument('--no-exclude', '-nx', dest='exclude_outliers', action='store_false')
    parser.set_defaults(exclude_outliers=False)

    parser.add_argument(
        "--save-root", "-r", type=str, default=SAVE_ROOT
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--jitter", "-j", type=float, default=0.1
    )

    return parser


def load_variant(
    variant,
    ckpt,
    is_eval=True, is_gt=True,
    eval_stat_root = '/nethome/jye72/share/objectnav_detailed/',
    **kwargs
):
    path = get_variant_path(variant, ckpt, is_eval=is_eval, is_gt=is_gt, eval_stat_root=eval_stat_root, **kwargs)
    data = torch.load(path)
    print(f'Loaded {path}: {data["step_id"]:.3g} frames')

    all_info = []
    for ep in data['payload']:
        stats = ep['stats']
        info = ep['info']
        ep_info = info['episode_info']
        activations = info['internal_activations']
        all_info.append({
            "ep": int(ep_info['episode_id']),
            "scene": ep_info['scene_id'].split("/")[-2],
            "success": int(stats['success']),
            "spl": float(stats['spl']),
            "d2g": float(stats['distance_to_goal']),
            "geodesic": float(ep_info['info']['geodesic_distance']),
            "sge": float(stats['goal_vis']),
            "visit_count": float(stats['coverage.visit_count']),
            "obj_cat": ep_info['object_category'],
            "obj_cat_num": task_cat2mpcat40_labels.index(ep_info['object_category']),
            "did_stop": ep['did_stop'],
            "coverage": float(stats['coverage.reached']),
            "steps": stats['coverage.step'],
            "collisions": stats['collisions.count'],
            "weights": info['weights'], # T x K
            "actions": info['actions'], # T
            "gps": info['observations']['gps'].float(), # T x 3
            "compass": info['observations']['compass'], # T x 1
            "beliefs": activations['beliefs'].float(), # T x L x K X H?
            "fused_belief": activations['fused_belief'].float(),
            "fused_obs": activations['fused_obs'].float(),
            "action_logits": activations['action_logits'],
            "critic_values": activations['critic_values'],
            "timesteps": torch.arange(len(info['actions'])),
            "d2g_t": info['d2g'], # T
            "collisions_t": info['collisions_t'], # T
            "coverage_mini_t": info['coverage_t']['mini_reached'], # T
            "coverage_t": info['coverage_t']['reached'], # T
            "sge_t": info['sge_t'], # T
            "visit_count_t": info['visit_count'], # T
            "room_cat": info['room_cat']['room_cat'].to(torch.int), # T
        })
        if len(all_info[-1]['compass']) != all_info[-1]['steps'] - 1:
            print(
                ep_info['episode_id'],
                all_info[-1]['scene'],
                info['actions'].size(), info['weights'].size(),
                all_info[-1]['gps'].size(), all_info[-1]['compass'].size(),
                all_info[-1]['action_logits'].size(), all_info[-1]['critic_values'].size(),
                all_info[-1]['fused_obs'].size(), all_info[-1]['fused_belief'].size(),
                all_info[-1]['sge_t'].size(), all_info[-1]['steps'],
            )
            raise Exception('mismatch')
        #     print(f"{len(all_info[-1]['compass'])} {all_info[-1]['scene']} {ep_info['episode_id']}")

    df = pd.DataFrame(all_info)
    print(df[['obj_cat', 'scene', 'ep', 'success', 'steps', 'spl']])
    print('Success: ', df['success'].mean())
    print('SPL: ', df['spl'].mean())
    return df

def load_recurrent_model(variant, ckpt, index=0):
    # index: belief_index
    model = torch.nn.GRU(input_size=233, hidden_size=196, num_layers=1)
    pth = f"/srv/share/jye72/objectnav/{variant}/{variant}.{ckpt}.pth"
    weights = torch.load(pth, map_location='cpu')['state_dict']
    belief_prefix = f'actor_critic.net.state_encoders.{index}'
    rnn_weights = {
        k.split('.')[-1]: v for k, v in weights.items() if belief_prefix in k
    }
    model.load_state_dict(rnn_weights)
    return model

def extract_flat_key(df, key, index=-1):
    all_tensors = []
    ids = []
    for i in range(len(df)):
        item = df.iloc[i][key]
        if torch.is_tensor(item) and len(item.size()) >= 1:
            if index >= 0:
                item = item[:, index]
            all_tensors.append(item)
        else:
            all_tensors.append(torch.tensor([item] * int(df.iloc[i]['steps']-1)))
        ids.extend([i] * len(all_tensors[-1]))
    return torch.cat(all_tensors, dim=0).float(), ids

def get_model_inputs(df, scene='all', seed=0, index=0, size=1000):
    # TODO support for fused
    np.random.seed(seed)
    if scene == 'all':
        conditions = df
    else:
        conditions = df[df['scene'] == scene]
    obs, _ = extract_flat_key(conditions, 'fused_obs')
    beliefs, ep_ids = extract_flat_key(conditions, 'beliefs')
    ics = beliefs[:, index]

    choices = np.random.choice(np.arange(ics.size(0)), size=size)
    seeds = ics[choices]
    ep_ids = np.array(ep_ids)[choices]
    return obs.mean(dim=0), seeds, ep_ids, choices # technically ep_ilocs

def noise_ics(ics, seed=0, jitter_scale=0.01):
    np.random.seed(seed)
    jitter_energy = (ics ** 2).mean() * jitter_scale
    noise = (torch.rand_like(ics) - 0.5) * jitter_energy
    return ics + noise

def find_fixed_points(
    ics, obs, model,
    q_tol=1e-8,
    dq_tol=5e-8, # min improvement over best to reset patience
    dq_patience_tol=5e4, # 50K
    max_iters=5e6,
    device=None,
):
    if device is None:
        device = load_device()
    ics = ics.to(device)
    model = model.to(device)
    obs = obs.to(device)
    # Value quoted from Niru's line attractor paper
    # IC - batch of seeds for FP opt
    # Ref: https://github.com/mattgolub/fixed-point-finder/blob/master/FixedPointFinder.py
    # Unsqueeze for GRU layers * directions (dim=0)

    # 1. weight decay is bad and was hurting opt previously.
    # yet to see how it affected results.

    print(f"starting optimization with {ics.size(0)} points")
    states = nn.Parameter(ics.clone().to(device).unsqueeze(0))
    optimizer = optim.AdamW(
        [states],
        lr=1e-3,
        weight_decay=0.0,
        eps=1e-8,
    )

    # Adaptive training dropped as it appears to be less stable (n=1)
    # alr = AdaptiveLearningRate()
    # agnc = AdaptiveGradNormClip()

    full_obs = obs.unsqueeze(0).expand(ics.size(0), obs.size(-1)).unsqueeze(0)
    loss = 100.0
    iters = 0
    dq_patience = 0
    best_loss = loss
    while loss > q_tol and iters < max_iters:
        # iter_lr = alr()
        # iter_gnc = agnc()

        _, next_states = model(full_obs, states)
        deltas = ((next_states - states) ** 2).sum(dim=-1)
        loss = deltas.mean()
        optimizer.zero_grad()
        loss.backward()

        # for g in optimizer.param_groups:
        #     g['lr'] = iter_lr
        # grad_norm = torch.nn.utils.clip_grad_norm_(states, iter_gnc)

        optimizer.step()
        if loss.item() < best_loss - dq_tol:
            best_loss = loss.item()
            dq_patience = 0
        else:
            dq_patience += 1
            if dq_patience > dq_patience_tol:
                print("Stopping due to patience")
                break

        # alr.update(loss.cpu().detach().numpy())
        # agnc.update(grad_norm.cpu().detach().numpy())

        iters += 1
        if iters % 1000 == 0:
            print(f"Iter: {iters} \t dq patience: {dq_patience} \t loss: {loss.item()}")
    return next_states.squeeze(0).detach(), deltas[0].detach()

def find_outliers(
    source, subjects, mode='mean', threshold=10.0
):
    # This could presumably become an optimized term.
    if mode == 'nn': # distance to closest source node
        raise NotImplementedError
    if mode == 'mean': # distance to source centroid (from FP finder)
        centroid = source.mean(dim=0)
        source_dists = torch.linalg.norm(source - centroid, dim=-1)
        avg_dist = source_dists.mean()
        subject_dists = torch.linalg.norm(subjects - centroid, dim=-1)
        return subject_dists > avg_dist * threshold

def get_unique(fps, deltas, tolerance=6):
    # tolerance is order of magnitude
    # returns fps unique up to tolerance, returning the smallest delta among duplicates
    uniq_fps, inv = torch.unique(np.around(fps.cpu(), tolerance), dim=0, return_inverse=True)
    rebuilt_fps = []
    rebuilt_deltas = []
    for i in range(len(uniq_fps)):
        pool = fps[inv == i]
        pool_of_deltas = deltas[inv == i]
        best_of_pool = torch.argmin(pool_of_deltas)
        rebuilt_fps.append(pool[best_of_pool])
        rebuilt_deltas.append(pool_of_deltas[best_of_pool])
    return torch.stack(rebuilt_fps), torch.stack(rebuilt_deltas)

def run_fp_finder(
    variant='split-curric', ckpt=35, index=0, is_eval=True, tag='',
    num_fp=20000,
    scene='all',
    seed=0,
    zero_sge=False,
    override=False,
    save_root=SAVE_ROOT,
    jitter=0.01,
    exclude_outliers=False,
):
    cache_fps = get_cache_path(variant, ckpt, num_fp, index, scene, is_eval=is_eval, tag=tag)

    if not override and osp.exists(cache_fps):
        print(f"{cache_fps} exists, quitting.")
        return

    model = load_recurrent_model(variant, ckpt, index=index)
    df = load_variant(variant, ckpt, is_eval=is_eval)
    obs, ics, ids, choices = get_model_inputs(
        df, scene=scene, index=index, size=num_fp, seed=seed
    )
    if zero_sge:
        obs[-1] = 0
    ics = noise_ics(ics, seed=seed, jitter_scale=jitter)
    fps, deltas = find_fixed_points(ics, obs, model)

    if exclude_outliers:
        outlier_indices = find_outliers(ics, fps.cpu(), mode='mean', threshold=3.0)
        fps = fps[~outlier_indices]
        deltas = deltas[~outlier_indices]
        ids = np.array(ids)[~outlier_indices]
        choices = np.array(choices)[~outlier_indices]

    uniq_fps, deltas = get_unique(fps, deltas)
    print(f"{len(uniq_fps)} unique FPs in {len(fps)} total.")

    uniq_fps = uniq_fps.cpu()
    deltas = deltas.cpu()
    torch.save({
        'fps': uniq_fps,
        'ids': ids, # tracks ids sourcing the FPs. Probably not actually very informative.
        'choices': choices,
        'deltas': deltas,
        'source': ics # kept for outlier calculation
    }, cache_fps)
    print(f"Finished. Saved at {cache_fps}")

def get_cache_path(
    variant, ckpt, num_fp, index, scene,
    tag="",
    is_eval=True,
    save_root=SAVE_ROOT,
):
    return osp.join(save_root, f"{tag}_eval-{is_eval}_{variant}_{ckpt}_{num_fp}_{index}_{scene}.pth")

# Credit: https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
def get_batch_jacobian(model, state, inputs, state_grad=True, input_grad=False):
    # ? Not using this for now because I don't know why we tile `n_outputs` in size -2
    # state: b x h
    # inputs: embed_size (233)
    n, n_outputs = state.size()
    state = state.unsqueeze(1) # b x 1 x h...
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    state = state.repeat(1, n_outputs, 1) # 1 x b x h ? Why do we need n_outputs?
    inputs = inputs.repeat(1, n_outputs, 1) # 1 x b x h
    state.requires_grad_(state_grad)
    inputs.requires_grad_(input_grad) # Nah, we don't need this rn
    y = model(inputs, state) # Note, state needs 1 x b x h but we currently have b x n x h?
    input_val = torch.eye(n_outputs).reshape(1, n_outputs, n_outputs).repeat(n, 1, 1)
    y.backward(input_val)
    return state.grad.data, inputs.grad.data

def get_jac_pt(model, obs, states):
    # obs: h
    # states: b x h
    states = states.unsqueeze(0) # b x h -> 1 x b x h
    full_obs = obs.unsqueeze(0).expand(states.size(1), obs.size(-1)).unsqueeze(0)

    _, state_jac = jacobian(model.cpu(), (full_obs.cpu(), states.cpu()))
    obs_grad, state_grad = state_jac

    # Currently, instead of being b x h_out x h_in, we have 1 x b x h_out x 1 x b x h_in
    # This is just an implementation detail. Extract the block you need
    def extract_diag(grad):
        # grad: 1 x b x h_out x 1 x b x h_in
        # grad_out: b x h_out x h_in
        grad = grad.squeeze(3).squeeze(0) # b x h_out x b x h_in
        grad_pieces = []
        for i in range(grad.size(0)):
            grad_pieces.append(grad[i,:,i])
        return torch.stack(grad_pieces, dim=0)
    return extract_diag(obs_grad).cpu(), extract_diag(state_grad).cpu()

def get_jac_pt_sequential(model, obs, states):
    # No difference in time. This might be more memory efficient.
    # obs: h
    # states: b x h
    model = model.cpu()
    states = states.cpu()
    full_obs = obs.unsqueeze(0).cpu()
    def get_jac_pt_single(model, single_obs, single_state):
        # obs: h
        # state: h
        _, state_jac = jacobian(model, (
            single_obs.unsqueeze(0),
            single_state.unsqueeze(0).unsqueeze(0)
        ))
        obs_grad, state_grad = state_jac
        # 1 x 1 x h_out x 1 x 1 x h_in
        return obs_grad.squeeze(), state_grad.squeeze()
    obs_grads, state_grads = zip(*[get_jac_pt_single(model, full_obs, state) for state in states])
    return torch.stack(obs_grads, dim=0), torch.stack(state_grads, dim=0)

def get_eigvals(square): # B x N x N
    e_vals_unsrt, e_vecs_unsrt = np.linalg.eig(square)
    mags_unsrt = np.abs(e_vals_unsrt) # shape (b, n,)
    sort_idx = np.argsort(mags_unsrt, axis=-1)[:,::-1]
    # For each FP, sort eigenvectors in decreasing eigenvalue magnitude order
    all_vals = []
    all_vecs = []
    for k in range(len(e_vecs_unsrt)):
        sort_idx_k = sort_idx[k]
        e_vals_k = e_vals_unsrt[k][sort_idx_k]
        e_vecs_k = e_vecs_unsrt[k][:, sort_idx_k]
        all_vals.append(e_vals_k)
        all_vecs.append(e_vecs_k)
    return np.stack(all_vals, 0), np.stack(all_vecs, 0)

def time_const(spectrum):
    return np.abs(1 / np.log(np.abs(spectrum)))



def main():
    parser = get_parser()
    args = parser.parse_args()
    run_fp_finder(**vars(args))

if __name__ == "__main__":
    main()