#!/bin/bash
#SBATCH -a 0-199
#SBATCH -p bosch_gpu-rtx2080,ml_gpu-rtx2080
#SBATCH -c 4
#SBATCH -o ./cluster_logs/deepens_rs/%A-%a.o
#SBATCH -e ./cluster_logs/deepens_rs/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J nes-rs-esa # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION, on gpu $SLURMD_NODENAME"

# run this using the following loop: 
# for arch_id in $(cat < experiments/tiny/outputs/deepens_rs/run_1/incumbents.txt); do sbatch -p alldlc_gpu-rtx2080 cluster_scripts/tiny/sbatch_scripts/deepens_rs.sh $arch_id 1; done

# Activate virtual environment
source activate python36

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_deepens_baselearner.py --arch_id 7 --seed_id $SLURM_ARRAY_TASK_ID --working_directory "experiments/tiny/baselearners/nes_rs_esa/" --dataset tiny --num_epochs 100 --scheme nes_rs_esa --arch_path "experiments/tiny/baselearners/nes_rs/run_3/random_archs" --global_seed 3 --batch_size 128 --n_layers 8 --init_channels 36 --grad_clip --lr 0.1 --scheduler cosine


# Done
echo "DONE"
echo "Finished at $(date)"
