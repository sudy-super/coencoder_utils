export HF_HOME=~/
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG="TRACE"
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker,virbr,vmnet,vboxnet,eth0
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_SHM_DISABLE=0
export MASTER_ADDR=10.1.201.17
export MASTER_PORT=29500
export NCCL_DEBUG_FILE="/tmp/nccl_debug_rank_%r.log"

deepspeed --hostfile=hostfile --num_nodes=4 --num_gpus=8 finetune_mn.py --deepspeed ds_config_mn.json --pipeline-model-parallel-size 32