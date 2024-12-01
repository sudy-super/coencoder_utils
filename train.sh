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

torchrun --nnodes=4 --nproc_per_node=8 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=10.1.201.17:29500 finetune_mn.py