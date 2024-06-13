ts=$(date +%Y-%m-%d_%H-%M-%S)
PROJECT_NAME=contrast_attend_diffuse
BASE_DIR="/home/users/nus/li.rl/scratch/intern_kavi/std_output/${PROJECT_NAME}_${ts}"
CODE_DIR="/home/users/nus/li.rl/scratch/intern_kavi/vis_dec_neurips/code"

NUM_GPUS=1
mkdir -p $BASE_DIR
JOB=`/opt/pbs/bin/qsub -V -q normal<<EOJ
#!/bin/bash 
#PBS -l select=1:ngpus=${NUM_GPUS}
#PBS -l walltime=24:00:00
#PBS -P 11001932 
#PBS -N ${PROJECT_NAME}
#PBS -e "${BASE_DIR}/stderr.txt"
#PBS -o "${BASE_DIR}/stdout.txt"
    module load miniforge3/23.10
    conda activate /home/users/nus/li.rl/.conda/envs/kavi-mindeye
    cd ${CODE_DIR}

    torchrun --standalone clip_unclip.py

EOJ`

echo "${PROJECT_NAME}"
echo "JobID = ${JOB} submitted"