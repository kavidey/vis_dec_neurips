ts=$(date +%Y-%m-%d_%H-%M-%S)
PROJECT_NAME=contrast_attend_diffuse
BASE_DIR="/home/users/nus/li.rl/scratch/intern_kavi/std_output/${PROJECT_NAME}_${ts}"
CODE_DIR="/home/users/nus/li.rl/scratch/intern_kavi/vis_dec_neurips/code"

NUM_GPUS=2
mkdir -p $BASE_DIR
JOB=`/opt/pbs/bin/qsub -V -q normal<<EOJ
#!/bin/bash 
#PBS -l select=1:ngpus=${NUM_GPUS}
#PBS -l walltime=24:00:00
#PBS -P 11001932 
#PBS -N ${PROJECT_NAME}
#PBS -e "${BASE_DIR}/stderr.txt"
#PBS -o "${BASE_DIR}/stdout.txt"
    source /home/users/nus/li.rl/scratch/intern_kavi/MindEyeV2/mind-eye-2/bin/activate
    cd ${CODE_DIR}

    torchrun clip_unclip.py

EOJ`

echo "${PROJECT_NAME}"
echo "JobID = ${JOB} submitted"