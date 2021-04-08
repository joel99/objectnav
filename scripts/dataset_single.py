#%%
# Extract single episodes into new datasets

from pathlib import Path
import shutil
import gzip
import json
import numpy as np

root_dir = Path('/srv/share/datasets/objectnav/mp3d/v1')
seed = 0
np.random.seed(seed)
split = 'val' # source
new_key = 'test'
source = root_dir.joinpath(split)
target = root_dir.joinpath(new_key)
# Copy everything first, then we'll reduce
if target.exists():
    # pass
    raise Exception('target directory exists')
else:
    shutil.copytree(source, target)

target = target.joinpath('content')
jsongzs = list(target.glob('*.json.gz'))
num_scenes = len([x for x in jsongzs if x.is_file()])
slice_scenes = [size // num_scenes] * num_scenes
for i in range(size % num_scenes):
    slice_scenes[i] += 1
#%%
for i, jsongz in enumerate(jsongzs):
    with gzip.open(jsongz, "rt") as f: # ref-ing `pointnav_dataset`
        data = json.loads(f.read())
    num_eps = len(data['episodes'])
    slice_eps = np.random.choice(np.arange(num_eps), size=slice_scenes[i], replace=False)
    new_eps = [data['episodes'][i] for i in slice_eps]
    # new_eps = list(filter(lambda i, ep: i in slice_eps, data['episodes']))
    data['episodes'] = new_eps
    with gzip.open(jsongz, 'wt') as f:
        json.dump(data, f)
