#!/bin/bash
#SBATCH --job-name="Milestone1_Unroll"
#SBATCH --output="Milestone1_Unroll.out"
#SBATCH --error="Milestone1_Unroll.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun ./m1_unroll 10000 > m1_unroll.out
# { time srun ./m1_unroll 10000 > m1_unroll.out; } 2> time.out

# srun ncu --set full -f -o analysis_file ./m1_unroll 10000 > unrollm3_ncu.out

