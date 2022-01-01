# CubicOneShotSGD

Start by downloading the datasets:
```
bash download_datasets.sh
```

Let's assume that you have a conda env named `torch` for pytorch.
You can generate the test plot `test.png` by running:
```
conda activate torch
python train.py --seed 1
```

## Running on SLURM

Scripts for generating and submitting a job array are also available (but not tested).

Simply generate the job array and then submit the tasks as folows:
```
sh generate_ja.sh
sh submit_ja.sh
```

If training doesn't take that long on a cpu, and you want to make use of multi-processing, try something like this:
```
sh generate_ja.sh | sed 1d | parallel {}
```
