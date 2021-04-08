#%%
# Move files around for rednet
from pathlib import Path
from shutil import move

detailed_paths = Path('/srv/flash1/jye72/share/objectnav_detailed')
eval_paths = Path('/srv/flash1/jye72/share/objectnav_eval')

KEY = 'gt_False.pth'
NEW_KEY = 'gt_False_21.pth'
for var_path in eval_paths.glob("*"):
# for var_path in detailed_paths.glob("*"):
    for ckpt in var_path.glob("*"):
        for val in ckpt.glob("*"):
            val = str(val)
            if KEY in val:
                new_path = val[:-len(KEY)] + NEW_KEY
                move(val, new_path)

