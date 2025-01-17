import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer

# --------------------------------------------------
# 目標のバケットごとのトレイン・バリデーション件数
# --------------------------------------------------
desired_distribution = {
    "Pile": {
        "0-1024":       { "train": 6060,  "val": 53   },
        "1025-2048":    { "train": 4332,  "val": 44   },
        "2049-4096":    { "train": 4100,  "val": 39   },
        "4097-8192":    { "train": 6155,  "val": 61   },
        "8193-16384":   { "train": 9980,  "val": 100  },
        "16385-32768":  { "train": 14293, "val": 158  },
        "32769-65536":  { "train": 3250,  "val": 33   },
        "65537-131072": { "train": 1823,  "val": 19   },
        "total_train": 49993,
        "total_val":   507,
        "total":       50500
    },
    "arxiv": {
        "0-1024":       { "train": 5,     "val": 1  },
        "1025-2048":    { "train": 5,     "val": 1  },
        "2049-4096":    { "train": 70,    "val": 1  },
        "4097-8192":    { "train": 317,   "val": 4  },
        "8193-16384":   { "train": 1636,  "val": 17 },
        "16385-32768":  { "train": 2634,  "val": 27 },
        "32769-65536":  { "train": 1840,  "val": 19 },
        "65537-131072": { "train": 3497,  "val": 26 },
        "total_train": 10004,
        "total_val":   96,
        "total":       10100
    },
    "RedPajama": {
        "0-1024":       { "train": 5314, "val": 54  },
        "1025-2048":    { "train": 6560, "val": 62  },
        "2049-4096":    { "train": 5518, "val": 56  },
        "4097-8192":    { "train": 5351, "val": 54  },
        "8193-16384":   { "train": 11684,"val": 118 },
        "16385-32768":  { "train": 4100, "val": 38  },
        "32769-65536":  { "train": 1087, "val": 11  },
        "65537-131072": { "train": 389,  "val": 4   },
        "total_train": 40003,
        "total_val":   397,
        "total":       40400
    },
    "grand_total_train": 100000,
    "grand_total_val":   1000,
    "grand_total":       101000
}

desired_distribution = {
  "Pile": {
    "0-1024":       { "train": 2168, "val": 54 },  # total=2222
    "1025-2048":    { "train": 1553, "val": 39 },  # total=1592
    "2049-4096":    { "train": 1468, "val": 37 },  # total=1505
    "4097-8192":    { "train": 2206, "val": 56 },  # total=2262
    "8193-16384":   { "train": 3338, "val": 81 },  # total=3419
    "16385-32768":  { "train": 4222, "val": 106 }, # total=4328
    "32769-65536":  { "train": 2760, "val": 70 },  # total=2830
    "65537-131072": { "train": 1795, "val": 47 },  # total=1842

    "total_train": 19510,
    "total_val":   490,
    "total":       20000
  },

  "arxiv": {
    "0-1024":       { "train": 5,   "val": 1 },    # total=6
    "1025-2048":    { "train": 5,   "val": 1 },    # total=6
    "2049-4096":    { "train": 69,  "val": 2 },    # total=71
    "4097-8192":    { "train": 313, "val": 8 },    # total=321
    "8193-16384":   { "train": 661, "val": 17 },   # total=678
    "16385-32768":  { "train": 1331,"val": 33 },   # total=1364
    "32769-65536":  { "train": 928, "val": 24 },   # total=952
    "65537-131072": { "train": 1563,"val": 39 },   # total=1602

    "total_train": 4875,
    "total_val":   125,
    "total":       5000
  },

  "RedPajama": {
    "0-1024":       { "train": 2057, "val": 51 },  # total=2108
    "1025-2048":    { "train": 2536, "val": 62 },  # total=2598
    "2049-4096":    { "train": 2135, "val": 54 },  # total=2189
    "4097-8192":    { "train": 1728, "val": 43 },  # total=1771
    "8193-16384":   { "train": 2248, "val": 57 },  # total=2305
    "16385-32768":  { "train": 3458, "val": 80 },  # total=3538
    "32769-65536":  { "train": 1070, "val": 28 },  # total=1098
    "65537-131072": { "train": 383,  "val": 10 },  # total=393

    "total_train": 15615,
    "total_val":   385,
    "total":       16000
  },

  "grand_total_train": 40000,
  "grand_total_val":   1000,
  "grand_total":       41000
}

desired_distribution = {
  "Pile": {
    "0-1024":       { "train": 1224, "val": 41 },  # total=1265
    "1025-2048":    { "train": 904,  "val": 31 },  # total=935
    "2049-4096":    { "train": 1220, "val": 41 },  # total=1261
    "4097-8192":    { "train": 1590, "val": 52 },  # total=1642
    "8193-16384":   { "train": 2776, "val": 92 },  # total=2868
    "16385-32768":  { "train": 3516, "val": 111 }, # total=3627
    "32769-65536":  { "train": 1994, "val": 66 },  # total=2060
    "65537-131072": { "train": 1782, "val": 60 },  # total=1842

    "total_train": 15006,
    "total_val":   494,
    "total":       15500
  },

  "arxiv": {
    "0-1024":       { "train": 0,    "val": 1 },    # total=1
    "1025-2048":    { "train": 0,    "val": 1 },    # total=1
    "2049-4096":    { "train": 17,   "val": 1 },    # total=18
    "4097-8192":    { "train": 80,   "val": 3 },    # total=83
    "8193-16384":   { "train": 416,  "val": 15 },   # total=431
    "16385-32768":  { "train": 671,  "val": 22 },   # total=693
    "32769-65536":  { "train": 468,  "val": 16 },   # total=484
    "65537-131072": { "train": 1344, "val": 45 },   # total=1389

    "total_train": 2996,
    "total_val":   104,
    "total":       3100
  },

  "RedPajama": {
    "0-1024":       { "train": 1584, "val": 51 },  # total=1635
    "1025-2048":    { "train": 1946, "val": 66 },  # total=2012
    "2049-4096":    { "train": 1643, "val": 54 },  # total=1697
    "4097-8192":    { "train": 1327, "val": 45 },  # total=1372
    "8193-16384":   { "train": 1675, "val": 56 },  # total=1731
    "16385-32768":  { "train": 2380, "val": 82 },  # total=2462
    "32769-65536":  { "train": 1063, "val": 35 },  # total=1098
    "65537-131072": { "train": 380,  "val": 13 },  # total=393

    "total_train": 11998,
    "total_val":   402,
    "total":       12400
  },

  "grand_total_train": 30000,
  "grand_total_val":   1000,
  "grand_total":       31000
}


# ------------------------------
# データセット名とファイルパス
# ------------------------------
dataset_paths = {
    "Pile":       "pile/pile_sub.jsonl",
    "arxiv":      "rp_arxiv/arxiv_doc_to_abs.jsonl",
    "RedPajama":  "rp_sub/rp_sub.jsonl",
}

# ------------------------------
# バケット一覧と下限～上限
# ------------------------------
bucket_ranges = [
    ("0-1024",       0,     1024),
    ("1025-2048",    1025,  2048),
    ("2049-4096",    2049,  4096),
    ("4097-8192",    4097,  8192),
    ("8193-16384",   8193,  16384),
    ("16385-32768",  16385, 32768),
    ("32769-65536",  32769, 65536),
    ("65537-131072", 65537, 131072),
]

# グローバル tokenizer
tokenizer = None

def init_tokenizer():
    """
    並列実行される各プロセスでの初期化。
    """
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=False)

def unify_format(json_obj):
    """
    "text" キーのみを使用し、それ以外のキーは破棄。
    """
    text = json_obj.get("text", "")
    if not text:
        return None
    return {"text": text}

def get_bucket_name(token_length):
    """
    トークン数が 131073以上なら None (破棄)。
    下限～上限に合う範囲のバケット名を返す。
    """
    if token_length >= 131073:
        return None
    for bucket_name, lower, upper in bucket_ranges:
        if lower <= token_length <= upper:
            return bucket_name
    return None

def reservoir_sample(item, bucket_info):
    """
    リザーバーサンプリング (メモリ削減用)
      bucket_info = {
        "items": [],
        "capacity": int,  # target_count (train + val) 合計
        "seen_count": int
      }
    """
    bucket_info["seen_count"] += 1
    c = bucket_info["capacity"]
    cur_len = len(bucket_info["items"])
    
    if cur_len < c:
        bucket_info["items"].append(item)
    else:
        r = random.randint(0, bucket_info["seen_count"] - 1)
        if r < c:
            bucket_info["items"][r] = item

def chunk_reader(file_path, chunk_size=10000):
    """
    大規模ファイルを chunk_size 行ごとに読み込むジェネレータ。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def process_chunk_in_child(chunk, bucket_capacities):
    """
    子プロセス側で1チャンクを処理。
    - JSON行読み込み
    - "text" のみ取り出し
    - 131073トークン以上破棄
    - バケットにリザーバーサンプリング

    戻り値: { bucket_name: { "items": [...], "capacity": X, "seen_count": Y }, ... }
    """
    global tokenizer
    if tokenizer is None:
        init_tokenizer()
    
    # チャンク内局所リザーバー
    partial_reservoir = {}
    for bname, cap in bucket_capacities.items():
        partial_reservoir[bname] = {
            "items": [],
            "capacity": cap,
            "seen_count": 0
        }
    
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        try:
            raw_obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        data = unify_format(raw_obj)
        if data is None:
            continue
        
        token_length = len(tokenizer.tokenize(data["text"]))
        bname = get_bucket_name(token_length)
        if bname is None:
            continue

        reservoir_sample(data, partial_reservoir[bname])
    
    return partial_reservoir

def merge_reservoirs(global_reservoir, partial_reservoir):
    """
    親プロセス側でチャンクごとの部分リザーバーをグローバルリザーバーに取り込む。
    """
    for bname, pinfo in partial_reservoir.items():
        for item in pinfo["items"]:
            reservoir_sample(item, global_reservoir[bname])

def check_distribution(result_summary, desired_distribution):
    """
    result_summary と desired_distribution を比較し、警告または一致メッセージを出力する。
    """
    all_matched = True
    
    # データセットごとに比較
    for ds_name in ["Pile", "arxiv", "RedPajama"]:
        if ds_name not in result_summary or ds_name not in desired_distribution:
            print(f"[Warning] Distribution for {ds_name} not found in either result or desired.")
            all_matched = False
            continue
        for bname, tv_dict in desired_distribution[ds_name].items():
            if bname.startswith("total"):
                # "total_*" 系はあとでまとめてチェック
                continue
            desired_train = tv_dict["train"]
            desired_val   = tv_dict["val"]

            actual_train = result_summary[ds_name].get(bname, {}).get("train", 0)
            actual_val   = result_summary[ds_name].get(bname, {}).get("val", 0)

            if desired_train != actual_train:
                print(f"[Warning] {ds_name}, bucket={bname}, train mismatch: desired={desired_train}, got={actual_train}")
                all_matched = False
            if desired_val != actual_val:
                print(f"[Warning] {ds_name}, bucket={bname}, val mismatch: desired={desired_val}, got={actual_val}")
                all_matched = False

        # totals
        ds_desired_train = desired_distribution[ds_name].get("total_train", 0)
        ds_desired_val   = desired_distribution[ds_name].get("total_val", 0)
        ds_desired_total = desired_distribution[ds_name].get("total", 0)

        ds_actual_train  = result_summary[ds_name].get("total_train", 0)
        ds_actual_val    = result_summary[ds_name].get("total_val", 0)
        ds_actual_total  = result_summary[ds_name].get("total", 0)

        if ds_desired_train != ds_actual_train:
            print(f"[Warning] {ds_name} total_train mismatch: desired={ds_desired_train}, got={ds_actual_train}")
            all_matched = False
        if ds_desired_val != ds_actual_val:
            print(f"[Warning] {ds_name} total_val mismatch: desired={ds_desired_val}, got={ds_actual_val}")
            all_matched = False
        if ds_desired_total != ds_actual_total:
            print(f"[Warning] {ds_name} total mismatch: desired={ds_desired_total}, got={ds_actual_total}")
            all_matched = False

    # grand totals
    desired_gt_train = desired_distribution.get("grand_total_train", 0)
    desired_gt_val   = desired_distribution.get("grand_total_val", 0)
    desired_gt       = desired_distribution.get("grand_total", 0)

    actual_gt_train  = result_summary.get("grand_total_train", 0)
    actual_gt_val    = result_summary.get("grand_total_val", 0)
    actual_gt        = result_summary.get("grand_total", 0)

    if desired_gt_train != actual_gt_train:
        print(f"[Warning] grand_total_train mismatch: desired={desired_gt_train}, got={actual_gt_train}")
        all_matched = False
    if desired_gt_val != actual_gt_val:
        print(f"[Warning] grand_total_val mismatch: desired={desired_gt_val}, got={actual_gt_val}")
        all_matched = False
    if desired_gt != actual_gt:
        print(f"[Warning] grand_total mismatch: desired={desired_gt}, got={actual_gt}")
        all_matched = False

    if all_matched:
        print("[Info] Distribution perfectly matches the desired distribution!")
    else:
        print("[Info] Distribution does not match the desired distribution in some aspects.")

def main():
    random.seed(42)

    # ----------------------------
    # tokenizer初期化に関するログ
    # ----------------------------
    print("Initializing global tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=False)
    print("Tokenizer initialized.\n")

    # ----------------------------
    # max_workersのログ
    # ----------------------------
    max_workers = max(1, os.cpu_count() - 1)
    print(f"Using up to {max_workers} workers for parallel.\n")

    # --------------------------------------------------
    # まず、各データセットで (train+val) 合計が何件必要かを計算
    # --------------------------------------------------
    per_dataset_buckets = {}
    for ds_name, dist_info in desired_distribution.items():
        if ds_name in ["grand_total_train", "grand_total_val", "grand_total"]:
            continue
        if not isinstance(dist_info, dict):
            continue
        
        bucket_dict = {}
        for bname, tv_dict in dist_info.items():
            if bname.startswith("total"):
                continue
            train_need = tv_dict.get("train", 0)
            val_need   = tv_dict.get("val", 0)
            bucket_dict[bname] = train_need + val_need
        per_dataset_buckets[ds_name] = bucket_dict

    # --------------------------------------------------
    # グローバルリザーバー構造を作る (train+val 合計分)
    # --------------------------------------------------
    all_datasets_reservoirs = {}
    for ds_name, bucket_caps in per_dataset_buckets.items():
        reservoir_dict = {}
        for bname, cap in bucket_caps.items():
            reservoir_dict[bname] = {
                "items": [],
                "capacity": cap,
                "seen_count": 0
            }
        all_datasets_reservoirs[ds_name] = reservoir_dict

    chunk_size = 1000

    # --------------------------------------------------
    # 並列で各データセットを (train+val) 合計ぶんサンプリング
    # --------------------------------------------------
    for ds_name in per_dataset_buckets.keys():
        file_path = dataset_paths.get(ds_name, "")
        if not os.path.exists(file_path):
            print(f"[Warning] File not found for {ds_name}: {file_path}")
            continue
        
        # 行数をカウントしてプログレスバー設定
        total_lines = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
        num_chunks = (total_lines + chunk_size - 1) // chunk_size
        
        print(f"Processing dataset '{ds_name}' -> {file_path}")
        print(f"  {total_lines} lines, ~{num_chunks} chunks needed.")
        
        bucket_res = all_datasets_reservoirs[ds_name]
        bucket_caps = {b: r["capacity"] for b, r in bucket_res.items()}

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_tokenizer) as executor:
            with tqdm(total=num_chunks, desc=f"[{ds_name}] chunks") as pbar:
                for chunk in chunk_reader(file_path, chunk_size=chunk_size):
                    fut = executor.submit(process_chunk_in_child, chunk, bucket_caps)
                    futures.append(fut)

                    # 適度に回収してメモリ節約
                    if len(futures) >= max_workers * 2:
                        for fut in as_completed(futures):
                            partial_res = fut.result()
                            merge_reservoirs(bucket_res, partial_res)
                        futures = []
                        pbar.update(max_workers * 2 // 1)
                
                # 残りフューチャーを回収
                for fut in as_completed(futures):
                    partial_res = fut.result()
                    merge_reservoirs(bucket_res, partial_res)
                    pbar.update(1)

        print(f"Done dataset '{ds_name}'.\n")

    # --------------------------------------------------
    # 取得したサンプルをランダム分割して (train/val) に仕分け
    # --------------------------------------------------
    train_list = []
    val_list = []

    # 集計用: {ds_name: { bname: {"train":X, "val":Y}, "total_train":..., "total_val":..., "total":... }, ...}
    result_summary = {}

    for ds_name, reservoir_dict in all_datasets_reservoirs.items():
        ds_result = {}
        ds_result["total_train"] = 0
        ds_result["total_val"] = 0
        ds_result["total"] = 0

        for bname, info in reservoir_dict.items():
            items = info["items"]
            random.shuffle(items)

            train_need = desired_distribution[ds_name][bname]["train"]
            val_need   = desired_distribution[ds_name][bname]["val"]
            total_need = train_need + val_need

            actual_count = len(items)
            if actual_count < total_need:
                print(f"[Warning] {ds_name} bucket '{bname}': required={total_need}, got={actual_count}")

            val_part   = items[:val_need]
            train_part = items[val_need:val_need + train_need]

            ds_result.setdefault(bname, {})
            ds_result[bname]["train"] = len(train_part)
            ds_result[bname]["val"]   = len(val_part)

            ds_result["total_train"] += len(train_part)
            ds_result["total_val"]   += len(val_part)
            ds_result["total"]       += (len(train_part) + len(val_part))

            for d in train_part:
                d["source_dataset"] = ds_name
                train_list.append(d)
            for d in val_part:
                d["source_dataset"] = ds_name
                val_list.append(d)

        result_summary[ds_name] = ds_result

    # grand total
    grand_train = 0
    grand_val   = 0
    grand_total = 0
    for ds_name, ds_res in result_summary.items():
        grand_train += ds_res["total_train"]
        grand_val   += ds_res["total_val"]
        grand_total += ds_res["total"]

    result_summary["grand_total_train"] = grand_train
    result_summary["grand_total_val"]   = grand_val
    result_summary["grand_total"]       = grand_total

    # --------------------------------------------------
    # train.jsonl / val.jsonl に書き出し
    # --------------------------------------------------
    # random.shuffle(train_list)
    # random.shuffle(val_list)

    print(f"Writing train set: {len(train_list)} lines -> train.jsonl")
    with open("train.jsonl", "w", encoding="utf-8") as f:
        for item in train_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Writing val set: {len(val_list)} lines -> val.jsonl")
    with open("val.jsonl", "w", encoding="utf-8") as f:
        for item in val_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # --------------------------------------------------
    # 集計レポートを JSON で保存
    # --------------------------------------------------
    print("Writing distribution report -> distribution_report.json")
    with open("distribution_report.json", "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------
    # 分布を比較し、警告 or 一致メッセージを出力
    # --------------------------------------------------
    check_distribution(result_summary, desired_distribution)
    
    print("All done.")

if __name__ == "__main__":
    main()