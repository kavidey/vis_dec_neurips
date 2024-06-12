ts=$(date +%Y-%m-%d_%H-%M-%S)
PROJECT_NAME=vis-dec
BASE_DIR="/home/internkavi/kavi_tmp/std_outputs/${PROJECT_NAME}_${ts}"
CODE_DIR="/home/internkavi/kavi_tmp/vis_dec_neurips/code"
NUM_GPUS=1
mkdir -p $BASE_DIR
JOB=`/opt/pbs/bin/qsub -k oe -V -q gpuQ<<EOJ
#!/bin/bash
#PBS -N ${PROJECT_NAME}
#PBS -l walltime=100:00:00
#PBS -l select=1:ncpus=4:ngpus=${NUM_GPUS}:mem=100gb
#PBS -e "${BASE_DIR}/stderr.txt"
#PBS -o "${BASE_DIR}/stdout.txt"
    source activate /home/internkavi/miniconda3/envs/mind-eye-2
    module load cuda/11.7
    cd ${CODE_DIR}
    python ldm_condition.py
EOJ`
echo "${PROJECT_NAME}"
echo "JobID = ${JOB} submitted"
