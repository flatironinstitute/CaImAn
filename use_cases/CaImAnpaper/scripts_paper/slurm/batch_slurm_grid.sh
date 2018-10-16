#!/bin/bash
# Run 21 tasks, with ids 0..20. Have seven on a node 
# The stdout (-o) and stderr (-e) files will be written to the submission directory with
# names that follow the given templates (in this case, the names will include the
# SLURM job identifier).
#SBATCH -p ib -N9 --ntasks-per-node=1 --exclusive -o sbExample.%j.out -e sbExample.%j.err

# srun starts a process per task, 21 in this case. In the first example, we didn't explicitly set
# a number of tasks, so srun assumed there would be one per node.
srun bash -c 'KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=-1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/home/agiovann/anaconda3/envs/caiman_0918/bin/python /mnt/home/agiovann//SOFTWARE/CaImAn/use_cases/CaImAnpaper/scripts_paper/batch_grid_params_search.py $SLURM_PROCID'
