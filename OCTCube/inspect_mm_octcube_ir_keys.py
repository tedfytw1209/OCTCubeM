# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.

"""Pre-flight inspection of the OCTCube-IR joint checkpoint (mm_octcube_ir.pt).

`main_finetune_downstream_UFcohort_OCTCubeIR.py` assumes the checkpoint's
state_dict prefixes the 3D OCT tower with `visual.` and the 2D en-face tower with
`text.` (OpenCLIP naming convention, see retinal-COEM). This script loads the
checkpoint and prints the top-level key-prefix counts so that assumption can be
confirmed before launching the (long) HiPerGator sweep.

Usage:
    python inspect_mm_octcube_ir_keys.py /blue/ruogu.fang/tienyuchang/OCTCubeM/ckpt/mm_octcube_ir.pt
"""

import sys
from collections import Counter

import torch


def main(path):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu')

    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys()))
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            print("Using ckpt['state_dict']")
        elif 'model' in ckpt:
            sd = ckpt['model']
            print("Using ckpt['model']")
        else:
            sd = ckpt
            print("Using the checkpoint dict itself as the state_dict")
        if 'epoch' in ckpt:
            print("epoch:", ckpt['epoch'])
    else:
        sd = ckpt
        print("Checkpoint is a bare state_dict")

    # strip a leading DDP 'module.' prefix (as the loader does)
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

    total = len(sd)
    first_level = Counter(k.split('.', 1)[0] for k in sd.keys())
    print(f"\nTotal parameters: {total}")
    print("First-level prefix counts:")
    for prefix, count in sorted(first_level.items(), key=lambda x: -x[1]):
        print(f"  {prefix:<20s} {count}")

    n_visual = sum(1 for k in sd if k.startswith('visual.'))
    n_text = sum(1 for k in sd if k.startswith('text.'))
    print(f"\nvisual.* (expected OCT tower)   : {n_visual}")
    print(f"text.*   (expected en-face tower): {n_text}")

    if n_visual == 0 or n_text == 0:
        print("\n[WARNING] Expected non-empty 'visual.*' and 'text.*' groups. "
              "The tower prefixes may differ from the assumption in "
              "main_finetune_downstream_UFcohort_OCTCubeIR.py; adjust the split "
              "prefixes there accordingly.")
    else:
        print("\n[OK] Both 'visual.*' and 'text.*' groups are present. "
              "The OCTCubeIR loader's prefix split should work.")

    # show a few example keys from each group to sanity-check submodule names
    def sample(prefix, n=8):
        keys = [k for k in sd if k.startswith(prefix)]
        return keys[:n]

    print("\nExample visual.* keys:")
    for k in sample('visual.'):
        print("   ", k, tuple(sd[k].shape))
    print("Example text.* keys:")
    for k in sample('text.'):
        print("   ", k, tuple(sd[k].shape))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
