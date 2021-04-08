#!/srv/flash1/jye72/anaconda3/envs/habitat2/bin/python

# Simple script to check whether there are checkpoints to be evaled for a variant (and runs them)
# To be run from project root

import os
import os.path as osp
import sys
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant", "-v",
        type=str,
        required=True,
        help="name of variant",
    )

    parser.add_argument(
        "--suffix", "-s",
        type=str,
        required=True,
        help="variant suffix"
    )

    parser.add_argument(
        "--ckpt", "-c",
        type=int,
        required=True,
        help="earliest checkpoint to check"
    )

    parser.add_argument('--gt-semantics', '-gt', dest='gt_semantics', action='store_true')
    parser.add_argument('--no-gt-semantics', '-ngt', dest='gt_semantics', action='store_false')
    parser.set_defaults(gt_semantics=False)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    # return # ! We're using too many gpus, pause this

    # variant = sys.argv[1]
    # suffix = sys.argv[2]
    # ckpt = int(sys.argv[3])
    # gt_semantics = False if len(sys.argv) <= 4 else sys.argv[4] == "True"
    # check(variant, suffix, ckpt, gt_semantics)
    check(**vars(args))

def launch(variant, suffix, ckpt, gt_semantics):
# def launch(variant: str, suffix: str, ckpt: int, gt_semantics: bool):
    cmd_str = f"/srv/flash1/jye72/projects/embodied-recall/scripts/eval_on.sh {variant} {suffix} {ckpt}"
    print("launching")
    if gt_semantics:
        cmd_str += " -gt"
    print(cmd_str)
    # os.system(f"echo {cmd_str}")
    os.system(f"sbatch -x calculon,cortana {cmd_str}")

def check(variant, suffix, ckpt, gt_semantics):
# def check(variant: str, suffix: str, ckpt: int, gt_semantics: bool):

    # Get available checkpoints
    ckpt_dir = f"/srv/share/jye72/objectnav/{variant}-{suffix}/"
    ckpts = []
    if not osp.exists(ckpt_dir):
        return
    ckpts = [int(p.split('.')[-2]) for p in os.listdir(ckpt_dir)]

    # Get existing evals
    existing_evals = []
    eval_dir = f"/srv/share/jye72/objectnav_eval/{variant}-{suffix}"
    if osp.exists(eval_dir):
        relevant_eval_file = f'eval_gt_{str(gt_semantics)}.json'
        for c in os.listdir(eval_dir):
            if osp.exists(osp.join(eval_dir, c, relevant_eval_file)):
                existing_evals.append(int(c))

    # Get pending evals from watchfile
    os.makedirs('/srv/flash1/jye72/projects/embodied-recall/watch', exist_ok=True)
    watchfile = f"/srv/flash1/jye72/projects/embodied-recall/watch/{variant}_{suffix}_{'gt' if gt_semantics else 'pred'}"
    pending_evals = []
    if osp.exists(watchfile):
        with open(watchfile, 'r') as f:
            pending_evals = [int(c) for c in f.readlines()]

    # Kick off and record the most recent checkpoints
    start = max([ckpt - 1, *pending_evals, *existing_evals]) + 1

    with open(watchfile, 'a') as f:
        for c in sorted(ckpts):
            if c >= start:
                f.write(f'{c}\n')
                launch(variant, suffix, c, gt_semantics)

if __name__ == "__main__":
    main()
