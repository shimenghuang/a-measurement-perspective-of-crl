# grid.py
from itertools import product
import json, pathlib

lrs              = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3] # 6
seeds            = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #10
n_steps        = [11, 21, 31, 51, 101, 501, 1001, 5001, 10001, 20001] # 10
bs            = [512, 1024, 2048, 4096] # 4
out = pathlib.Path("grid.txt").open("w")
for lr, s, step, batch_size in product(lrs, seeds, n_steps, bs):
    # build the command-line fragment
    args = f"--lr {lr} --seed {s} --n-steps {step} --batch-size {bs}"
    # you can prepend other static flags here once, e.g. --n-steps 50001
    print(args, file=out)
print(f"Wrote {sum(1 for _ in product(lrs, seeds, n_steps, bs))} lines.")
