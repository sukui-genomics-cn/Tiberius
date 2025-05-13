#!/bin/bash
#DSUB -n danio-rerio-chr25_tiberius-hmm-500k_0512
#DSUB -A root.project.P24Z10200N0983
#DSUB -R 'cpu=32;gpu=1;mem=100000'
#DSUB -N 1
#DSUB -eo /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/logs/danio.rerio.chr25_tiberius.hmm.500k_0512.%J.%I.err
#DSUB -oo /home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius/logs/danio.rerio.chr25_tiberius.hmm.500k_0512.%J.%I.out

### Load WC Env
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
source /home/share/huadjyin/home/s_sukui/envs/env_py38_torch17

source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate tf2_10
echo 'Conda environment activated: tf2_10.'

##Config NCCL
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1


## Set scripts
RANK_SCRIPT="./scripts/tasks/danio.rerio.chr25_tiberius.hmm.500k.sh"

###Set Start Path
JOB_PATH=/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Tiberius

## Set NNODES
NNODES=1
GPUS=1

## Create nodefile

JOB_ID=${BATCH_JOB_ID}
NODEFILE=${JOB_PATH}/outputs/tmp/${JOB_ID}.nodefile
# touch ${NODEFILE}
touch $NODEFILE
cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1,"slots="$2}' > ${JOB_PATH}/outputs/tmp/${JOB_ID}.nodefile
cat $CCS_ALLOC_FILE | grep ^cyclone001-agent | awk '{print $1}' > ${NODEFILE}
cat ${CCS_ALLOC_FILE} > :q!${JOB_PATH}/outputs/tmp/CCS_ALLOC_FILE

cd ${JOB_PATH};/usr/bin/bash ${RANK_SCRIPT} ${NNODES} ${GPUS} ${NODEFILE}
