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
export MASTER_ADDR=10.1.201.21
export MASTER_PORT=29500

deepspeed --hostfile=hostfile --num_nodes=3 --num_gpus=8 finetune_mn.py --deepspeed ds_config_mn.json