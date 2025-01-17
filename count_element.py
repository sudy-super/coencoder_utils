import json
from transformers import AutoTokenizer
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import io
import itertools

def count_lines_chunk(file_path, start, size, chunk_buffer_size=8192):
    """
    より効率的で安全な行数カウント関数
    chunk_buffer_size: 一度に読み込むバッファサイズ
    """
    count = 0
    last_chunk = None
    
    try:
        with open(file_path, 'rb') as f:
            f.seek(start)
            if start != 0:
                f.readline()
            
            bytes_remaining = min(size, os.path.getsize(file_path) - f.tell())
            
            while bytes_remaining > 0:
                read_size = min(chunk_buffer_size, bytes_remaining)
                current_chunk = f.read(read_size)
                
                if not current_chunk:
                    break
                    
                count += current_chunk.count(b'\n')
                bytes_remaining -= len(current_chunk)
                last_chunk = current_chunk
                
            if last_chunk is not None and not last_chunk.endswith(b'\n'):
                count += 1
                
        return count
    except Exception as e:
        print(f"Error counting lines in chunk at position {start}: {e}")
        return None

def safe_count_lines_parallel(file_path, num_workers=None):
    """
    エラー処理を強化した並列行数カウント関数
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)
    
    try:
        file_size = os.path.getsize(file_path)
        chunk_size = max(1024 * 1024, file_size // num_workers)
        
        chunks = []
        for i in range(0, file_size, chunk_size):
            size = min(chunk_size, file_size - i)
            chunks.append((i, size))
        
        total_lines = 0
        retries = 3
        
        while retries > 0 and chunks:
            failed_chunks = []
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(count_lines_chunk, file_path, start, size): (start, size)
                    for start, size in chunks
                }
                
                with tqdm(total=len(futures), desc=f"行数カウント (残りリトライ: {retries})") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                total_lines += result
                            else:
                                failed_chunks.append(futures[future])
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
                            failed_chunks.append(futures[future])
                        pbar.update(1)
            
            chunks = failed_chunks
            retries -= 1
            
            if failed_chunks:
                print(f"\n{len(failed_chunks)}個のチャンクが失敗しました。")
                if retries > 0:
                    print(f"残り{retries}回リトライします...")
                    time.sleep(1)
        
        if failed_chunks:
            print("\n警告: いくつかのチャンクの処理に失敗しました。行数は概算となります。")
        
        return total_lines
    
    except Exception as e:
        print(f"Error in parallel line counting: {e}")
        print("並列処理に失敗しました。通常の行数カウントを試みます...")
        try:
            with open(file_path, 'rb') as f:
                return sum(1 for _ in f)
        except Exception as e:
            print(f"通常の行数カウントも失敗しました: {e}")
            return None

def init_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", use_fast=False)

def process_lines(lines):
    """
    指定された行のリストを処理する関数
    """
    try:
        lengths = []
        for line in lines:
            try:
                data = json.loads(line)
                text = data["text"]
                length = len(tokenizer.tokenize(text))
                lengths.append(length)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
        return lengths
    except Exception as e:
        print(f"Error in process_lines: {e}")
        return []

def get_range_count(lengths):
    counter = {
        "131073~": 0,
        "65537~131072": 0,
        "32769~65536": 0,
        "16385~32768": 0,
        "8193~16384": 0,
        "4097~8192": 0,
        "2049~4096": 0,
        "1025~2048": 0,
        "0~1024": 0
    }
    
    for length in lengths:
        if length >= 131073:
            counter["131073~"] += 1
        elif length >= 65537:
            counter["65537~131072"] += 1
        elif length >= 32769:
            counter["32769~65536"] += 1
        elif length >= 16385:
            counter["16385~32768"] += 1
        elif length >= 8193:
            counter["8193~16384"] += 1
        elif length >= 4097:
            counter["4097~8192"] += 1
        elif length >= 2049:
            counter["2049~4096"] += 1
        elif length >= 1025:
            counter["1025~2048"] += 1
        else:
            counter["0~1024"] += 1
    
    return counter

class ChunkReader:
    def __init__(self, file_path, chunk_size=100):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.file = None
        
    def __enter__(self):
        self.file = open(self.file_path, 'r', encoding='utf-8')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            
    def __iter__(self):
        chunk = []
        for line in self.file:
            chunk.append(line)
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def merge_counters(counter1, counter2):
    return {k: counter1.get(k, 0) + counter2.get(k, 0) for k in counter1.keys()}

def main():
    start_time = time.time()
    file_path = "rp_sub/rp_sub.jsonl" # "rp_arxiv/arxiv_doc_to_abs.jsonl" # "pile/pile_sub.jsonl"
    
    print("トークナイザーを読み込んでいます...")
    init_tokenizer()
    print("トークナイザーの読み込みが完了しました")

    # プロセス数とチャンクサイズの設定
    max_workers = max(1, os.cpu_count() - 1)
    chunk_size = 1000  # より小さいチャンクサイズに設定
    
    # 安全な並列行数カウント
    print("ファイルの行数を並列カウントしています...")
    total_lines = safe_count_lines_parallel(file_path, max_workers)
    
    if total_lines is None:
        print("行数カウントに失敗しました。プログラムを終了します。")
        return
    
    print(f"合計行数: {total_lines}")
    total_chunks = (total_lines + chunk_size - 1) // chunk_size
    print(f"\n{max_workers}個のワーカーで処理を開始します...")
    
    # 結果を格納するカウンター
    total_counter = {
        "131073~": 0,
        "65537~131072": 0,
        "32769~65536": 0,
        "16385~32768": 0,
        "8193~16384": 0,
        "4097~8192": 0,
        "2049~4096": 0,
        "1025~2048": 0,
        "0~1024": 0
    }

    completed_chunks = 0
    with tqdm(total=total_chunks, desc="チャンク処理") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_tokenizer) as executor:
            with ChunkReader(file_path, chunk_size) as reader:
                # チャンクごとに処理を実行
                futures = []
                for chunk in reader:
                    futures.append(executor.submit(process_lines, chunk))
                    
                    # 一定数のフューチャーがたまったら処理
                    if len(futures) >= max_workers * 2:
                        for future in as_completed(futures):
                            try:
                                lengths = future.result()
                                chunk_counter = get_range_count(lengths)
                                total_counter = merge_counters(total_counter, chunk_counter)
                                completed_chunks += 1
                                pbar.update(1)
                            except Exception as e:
                                print(f"Error processing chunk result: {e}")
                        futures = []
                
                # 残りのフューチャーを処理
                for future in as_completed(futures):
                    try:
                        lengths = future.result()
                        chunk_counter = get_range_count(lengths)
                        total_counter = merge_counters(total_counter, chunk_counter)
                        completed_chunks += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing chunk result: {e}")

    # 結果の表示
    print("\n結果:")
    for range_name, count in total_counter.items():
        print(f"{range_name}: {count}")
    
    # 処理時間の表示
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n総処理時間: {total_time:.2f}秒")
    print(f"処理完了チャンク数: {completed_chunks}")

if __name__ == "__main__":
    main()