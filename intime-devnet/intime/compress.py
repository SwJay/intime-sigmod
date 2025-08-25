import struct
from collections import Counter
import random
import time
import sys
import pickle
# from pympler import asizeof
import csv
import os
import psutil
from contextlib import contextmanager
import tracemalloc


def timestamp_ms_encode(timestamp):
    return round(timestamp * 1000)

def timestamp_ms_decode(timestamp):
    return timestamp / 1000.0

def delta_encoding(timestamps, base_time):
    return [t - base_time for t in timestamps]

def split_seconds_milliseconds(timestamps):
    seconds = [t // 1000 for t in timestamps]
    milliseconds = [t % 1000 for t in timestamps]
    return seconds, milliseconds

def varint_encode(number):
    encoded = []
    while number > 0:
        encoded.append(number & 0x7F)
        number >>= 7
        if number > 0:
            encoded[-1] |= 0x80
    return bytes(encoded) if encoded else b'\x00'

def varint_decode(byte_stream):
    result = 0
    shift = 0
    bytes_read = 0
    for byte in byte_stream:
        bytes_read += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, bytes_read

def compress_timestamps(timestamps):
    if not timestamps:
        return b''
    
    base_time = min(timestamps)  # Use minimum timestamp as base time
    delta_times = delta_encoding(timestamps, base_time)
    
    # Compress timestamps
    compressed_deltas = b''.join(varint_encode(dt) for dt in delta_times)
    
    # Combine compressed data
    packed_data = struct.pack('!Q', base_time) + compressed_deltas
    
    return packed_data

def decompress_timestamps(packed_data):
    base_time = struct.unpack('!Q', packed_data[:8])[0]
    data = packed_data[8:]
    
    # Decompress timestamps
    timestamps = []
    i = 0
    while i < len(data):
        delta, bytes_read = varint_decode(data[i:])
        i += bytes_read
        timestamps.append(base_time + delta)
    
    return timestamps

def generate_uniform_timestamps(n, start_time=None, end_time=None):
    if start_time is None:
        start_time = time.time()
    if end_time is None:
        end_time = start_time + 12  # Default range is 12s

    return [random.uniform(start_time, end_time) for _ in range(n)]

def get_true_size(obj):
    return len(pickle.dumps(obj))

def test_size_time():
    # Create CSV file and write header
    csv_filename = "res/compress.csv"
    fieldnames = ["Number of Timestamps", "Original Size (bytes)", "Compressed Size (bytes)", "Compression Ratio", "Compression Time (s)", "Decompression Time (s)"]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Generate different numbers of uniformly distributed timestamps and test compression effect
        for n in [32, 64, 128, 256, 512]:
            print(f"\nTesting {n} timestamps:")
            
            # Initialize result accumulators
            total_original_size = 0
            total_compressed_size = 0
            total_compress_time = 0
            total_decompress_time = 0

            # Repeat 100 times
            for _ in range(100):
                original_timestamps = generate_uniform_timestamps(n)
                original_timestamps_ms = [timestamp_ms_encode(t) for t in original_timestamps]
                
                start_compress = time.time()
                compressed_ms = compress_timestamps(original_timestamps_ms)
                compress_time = time.time() - start_compress
                
                start_decompress = time.time()
                decompressed_ms = decompress_timestamps(compressed_ms)
                decompress_time = time.time() - start_decompress
                
                original_size = get_true_size(original_timestamps_ms)
                compressed_size = get_true_size(compressed_ms)
                
                # Accumulate results
                total_original_size += original_size
                total_compressed_size += compressed_size
                total_compress_time += compress_time
                total_decompress_time += decompress_time

            # Calculate average values
            avg_original_size = total_original_size / 100
            avg_compressed_size = total_compressed_size / 100
            avg_compress_time = total_compress_time / 100
            avg_decompress_time = total_decompress_time / 100
            avg_compression_ratio = avg_compressed_size / avg_original_size

            # Write average results to CSV file
            writer.writerow({
                "Number of Timestamps": n,
                "Original Size (bytes)": f"{avg_original_size:.2f}",
                "Compressed Size (bytes)": f"{avg_compressed_size:.2f}",
                "Compression Ratio": f"{avg_compression_ratio:.2%}",
                "Compression Time (s)": f"{avg_compress_time:.6f}",
                "Decompression Time (s)": f"{avg_decompress_time:.6f}"
            })
            
            # Print average results
            print("Compression results (average):")
            print(f"Original size: {avg_original_size:.2f} bytes")
            print(f"Compressed size: {avg_compressed_size:.2f} bytes")
            print(f"Compression ratio: {avg_compression_ratio:.2%}")
            print(f"Compression time: {avg_compress_time:.6f} seconds")
            print(f"Decompression time: {avg_decompress_time:.6f} seconds")

    print(f"\nResults have been saved to {os.path.abspath(csv_filename)}")

def measure_performance(func):
    """测量函数的 CPU 和内存使用情况的装饰器"""
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_cpu_percent = process.cpu_percent()
        
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_cpu_percent = process.cpu_percent()
        
        cpu_usage = end_cpu_percent - start_cpu_percent
        
        return result, cpu_usage, peak / 1024  # 转换为 KB
    return wrapper

def test_cpu_memory():
    # 创建 CSV 文件并写入表头
    csv_filename = "res/compress_usage.csv"
    fieldnames = ["Number of Timestamps", "Compression CPU Usage (%)", "Compression Memory Usage (KB)",
                  "Decompression CPU Usage (%)", "Decompression Memory Usage (KB)"]
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for n in [32, 64, 128, 256, 512]:
            print(f"\n测试 {n} 个时戳:")
            
            # 初始化结果累加器
            total_compress_cpu = 0
            total_compress_memory = 0
            total_decompress_cpu = 0
            total_decompress_memory = 0

            for _ in range(100):
                original_timestamps = generate_uniform_timestamps(n)
                original_timestamps_ms = [timestamp_ms_encode(t) for t in original_timestamps]
                
                @measure_performance
                def compress():
                    return compress_timestamps(original_timestamps_ms)
                
                compressed_ms, compress_cpu, compress_memory = compress()
                
                @measure_performance
                def decompress():
                    return decompress_timestamps(compressed_ms)
                
                decompressed_ms, decompress_cpu, decompress_memory = decompress()
                
                # 累加结果
                total_compress_cpu += compress_cpu
                total_compress_memory += compress_memory
                total_decompress_cpu += decompress_cpu
                total_decompress_memory += decompress_memory

            # 计算平均值
            avg_compress_cpu = total_compress_cpu / 100
            avg_compress_memory = total_compress_memory / 100
            avg_decompress_cpu = total_decompress_cpu / 100
            avg_decompress_memory = total_decompress_memory / 100

            # 将平均结果写入 CSV 文件
            writer.writerow({
                "Number of Timestamps": n,
                "Compression CPU Usage (%)": f"{avg_compress_cpu:.2f}",
                "Compression Memory Usage (KB)": f"{avg_compress_memory:.2f}",
                "Decompression CPU Usage (%)": f"{avg_decompress_cpu:.2f}",
                "Decompression Memory Usage (KB)": f"{avg_decompress_memory:.2f}"
            })
            
            # 打印平均结果
            print("能结果（平均值）:")
            print(f"压缩 CPU 使用率: {avg_compress_cpu:.2f}%")
            print(f"压缩内存使用: {avg_compress_memory:.2f} KB")
            print(f"解压 CPU 使用率: {avg_decompress_cpu:.2f}%")
            print(f"解压内存使用: {avg_decompress_memory:.2f} KB")

    print(f"\n结果已保存到 {os.path.abspath(csv_filename)}")


if __name__ == "__main__":
    os.makedirs("res", exist_ok=True)
    
    # test_size_time()
    # test_cpu_memory()

