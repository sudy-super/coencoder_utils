global_rank = 12

def get_node_rank(gpu_rank):
    return gpu_rank // 8  # 各ノードに8 GPUがあるため

def get_device_map_for_rank(gpu_rank):
    # GPUのglobal rankからノード番号を取得
    node_rank = get_node_rank(gpu_rank)
    
    # 各ノードが担当するGPUの範囲を定義
    gpu_ranges = {
        0: (0, 7),    # node 0: cuda:0-7
        1: (8, 15),   # node 1: cuda:8-15
        2: (16, 23),  # node 2: cuda:16-23
        3: (24, 31)   # node 3: cuda:24-31
    }
    
    start_gpu, end_gpu = gpu_ranges[node_rank]
    device_map = {}
    
    # 基本的なdevice mapを定義
    base_map = {
        'context_tower.tower.embed_tokens': 'cuda:0',
        'connector.dynamic_pooling': 'cuda:12',
        'connector.linear_1': 'cuda:13',
        'connector.act': 'cuda:14',
        'connector.linear_2': 'cuda:14',
        'language_model.model.embed_tokens': 'cuda:14',
        'language_model.model.norm': 'cuda:31',
        'language_model.lm_head': 'cuda:31'
    }
    
    # context_tower layers
    for i in range(24):
        gpu_idx = i // 2
        if start_gpu <= gpu_idx <= end_gpu:
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            device_map[f'context_tower.tower.layers.{i}'] = f'cuda:{gpu_idx}'
    
    # language_model layers
    for i in range(32):
        gpu_idx = i // 2 + 15  # 15から始まるため
        if start_gpu <= gpu_idx <= end_gpu:
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            device_map[f'language_model.model.layers.{i}'] = f'cuda:{gpu_idx}'
    
    # 基本コンポーネントの割り当て
    for key, device in base_map.items():
        gpu_idx = int(device.split(':')[1])
        if start_gpu <= gpu_idx <= end_gpu:
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            if gpu_idx >= 8:
                gpu_idx -= 8
            device_map[key] = f'cuda:{gpu_idx}'
    
    return device_map

# 現在のglobal rankに対応するdevice_mapを取得
device_map = get_device_map_for_rank(global_rank)

# ローカルのGPU番号を取得（0-7の範囲）
local_gpu_id = global_rank % 8
print(f"Global Rank {global_rank} (Node {get_node_rank(global_rank)}, Local GPU {local_gpu_id}) device map:", device_map)