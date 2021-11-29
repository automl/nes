#!/bin/bash
#SBATCH -a 0-399
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH -x dlcgpu15,dlcgpu02,dlcgpu42
#SBATCH -o ./cluster_logs/nes_rs_oneshot/%A-%a.o
#SBATCH -e ./cluster_logs/nes_rs_oneshot/%A-%a.e
#SBATCH --gres=gpu:1  # reserves GPUs
#SBATCH -J oneshot-nes # sets the job name. If not specified, the file name will be used as job name

# Info
echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION, on gpu $SLURMD_NODENAME"

# Activate virtual environment
#source venv/bin/activate
source activate python37

# Arrayjob
PYTHONPATH=$PWD python nes/optimizers/scripts/train_oneshot_bsl.py --arch_id $SLURM_ARRAY_TASK_ID --seed_id $SLURM_ARRAY_TASK_ID --working_directory "experiments-nips21/cifar10/baselearners/nes_rs_darts/" --dataset cifar10 --num_epochs 100 --scheme nes_rs_darts --arch_path "experiments-nips21/cifar10/baselearners/nes_rs/run_${1}/random_archs" --global_seed $1 --only_predict --oneshot --saved_model "/home/zelaa/playground/darts/cnn/search-darts_model-20210809-163644"

# Done
echo "DONE"
echo "Finished at $(date)"
