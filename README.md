Auxiliary Tasks and Exploration Enable ObjectNav
==============================
Joel Ye, Dhruv Batra, Abhishek Das, and Erik Wijmans.

Project Site: [https://joel99.github.io/objectnav/](https://joel99.github.io/objectnav/)

This repo is a code supplement for the paper "Auxiliary Tasks and Exploration Enable ObjectNav." The code is a fork of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) and borrows heavily from [habitat-pointnav-aux](https://github.com/joel99/habitat-pointnav-aux).

## Overview
The primary code contributions from the paper are located in:
- Auxiliary Tasks: `auxiliary_tasks.py` and `supervised_auxiliary_tasks.py`
- Exploration Reward: `coverage.py`
- Tether logic (denoted as "split" in code): `rollout_storage.py`, `ppo.py`, `multipolicy.py`
- Experimental Configurations: `habitat_baselines/config/objectnav/full/*.yaml`
- Misellaneous business logic: `ppo_trainer.py`
- Scripts and Analysis: `./scripts/*`

## Requirements
The code is known to be compatible with Python 3.6, Habitat Lab v0.1.5 and Habitat Sim v0.1.6 (headless). We recommend you use conda to initialize the code as follows:
1. Install [habitat-sim](https://github.com/facebookresearch/habitat-sim#installation). In the habitat-sim repo, run:
```bash
conda create -n <ENV_NAME> python=3.6 cmake=3.14.0
conda activate <ENV_NAME>
pip install -r requirements.txt
conda install habitat-sim=0.1.6 headless -c conda-forge -c aihabitat
```
Then, in *this* repo, run:
```bash
conda env update -f environment.yml --name <ENV_NAME>
```
Note: If you run into a pip error along the lines of a [missing metadata file](https://stackoverflow.com/questions/54552367/pip-cannot-find-metadata-file-environmenterror), it is likely an improperly cleaned package (e.g. numpy-1.19.5), just remove it.
Note 2: You may also need to run `python setup.py develop --all` as in the [habitat-lab installation process](https://github.com/facebookresearch/habitat-lab).

### Data
The experiments were done on the Matterport3D dataset. Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim#datasets) for the scene dataset and [habitat-lab](https://github.com/facebookresearch/habitat-lab#task-datasets) for the ObjectNav dataset.

### Pre-trained Weights
You can download pretrained agent and rednet weights [here](https://drive.google.com/drive/folders/1SM75RweHtHQ13lu9fZkVjkOlWMaWpFuZ?usp=sharing). We provide the best checkpoint for agents as measured by their performance with RedNet segmentation:
- `base-full.31.pth`: 6-action agent.
- `base4-full.33.pth`: 4-action agent.
- `split_clamp.31.pth`: 6 + Tether agent.
- `rednet_semmap_mp3d_tuned.pth`: RedNet weights.

## Training
We train our models for around 8 GPU-weeks, on a SLURM-managed cluster. On such a cluster, you can launch your run with:

> `sbat ./scripts/ddppo_8gpu.sh <args>`

With the appropriate configuration file. The code in `run.py` is set up to parse out the arguments, so to train with the configuration `.../objectnav/full/base.yaml`, write:

> `sbat ./scripts/ddppo_8gpu.sh base full`

On a machine without SLURM, see, e.g. `./scripts/train_no_ckpt.sh`.

## Evaluation
You can evaluate a trained checkpoint by configuring the arguments in `./scripts/eval_on.sh`. This forwards to `habitat_baselines/run.py`, which accepts a number of flags. For example:
- Add a `-v` flag to generate videos of your agent.
- Add a `-ra` flag to record trajectories and hidden states for subsequent analysis.

See `run.py` for all arguments. Note that an evaluation on the full validation split can take a while (>1 GPU-day).

## Citation
If you use this work, you can cite it as
```
@misc{ye2021auxiliary,
      title={Auxiliary Tasks and Exploration Enable ObjectNav},
      author={Joel Ye and Dhruv Batra and Abhishek Das and Erik Wijmans},
      year={2021},
      eprint={2104.04112},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
### Additional Notes
- The 300-episode subset mentioned throughout analysis was created by sampling episodes uniformly across the scenes in train and val splits, using `scripts/dataset_slicer.py`
- The `.py` notebooks in `scripts/` start with a comment denoting their use.
- Though the configurations default to mixed precision training, it is not particularly faster in our experiments.
- The UNLICENSE applies to this project's changes on top of the habitat-lab codebase.
