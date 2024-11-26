export HF_HOME=/workspace
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export MASTER_ADDR=10.1.201.02
export MASTER_PORT=29500

deepspeed --hostfile=hostfile --num_nodes=4 --num_gpus=8 finetune_new.py --deepspeed ds_config_new.json