export HF_HOME=~/
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="TRACE"
export NCCL_SOCKET_IFNAME=^lo,docker,virbr,vmnet,vboxnet,eth0
export NCCL_P2P_DISABLE=1
export MASTER_ADDR=10.1.201.02
export MASTER_PORT=29500

deepspeed --hostfile=hostfile --num_nodes=4 --num_gpus=8 finetune_mn.py --deepspeed ds_config_mn.json