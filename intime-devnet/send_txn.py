# send_transaction.py

from web3 import Web3, AsyncWeb3
import asyncio
import time
import pandas as pd
import numpy as np
from collections import deque
import websockets
from hexbytes import HexBytes
import sys
import os

LISTEN_URL = 'ws://localhost:8100'
BLOCK_NUM = 100
BASE_HEIGHT = 19000000

sender_address = Web3.to_checksum_address("0x123463a4b065722e99115d6c222f267d9cabb524")

# 全局变量用于存储统计信息
total_transactions = 0
transaction_times = []
start_time = None


# async def send_txn():
#     # Create txn
#     tx = {
#         'to': recipient_address,
#         'value': 1, #w3.to_wei(0.0001, 'ether'),
#         'from': sender_address,
#     }

#     # Send txn
#     tx_hash = await w3.eth.send_transaction(tx)

#     print(f"Tx sent: {tx_hash.hex()}")

    # Wait for txn receipt
    # tx_receipt = await w3.eth.wait_for_transaction_receipt(tx_hash)
    # print(f"Tx mined: {tx_hash.hex()}, with status:{'Success' if tx_receipt['status'] == 1 else 'Failed'}")


async def send_single_tx(tx):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            tx_start_time = time.time()
            tx_hash_result = await w3.eth.send_transaction(tx)
            
            # Handle different types of return values
            if isinstance(tx_hash_result, dict) and 'result' in tx_hash_result:
                tx_hash = tx_hash_result['result']
            elif isinstance(tx_hash_result, (str, HexBytes)):
                tx_hash = tx_hash_result
            else:
                raise ValueError(f"Unexpected transaction hash type: {type(tx_hash_result)}")
            
            # Ensure tx_hash is HexBytes type
            if not isinstance(tx_hash, HexBytes):
                tx_hash = HexBytes(tx_hash)
            
            return tx_hash, tx_start_time
        except Exception as e:
            print(f"Error sending transaction (attempt {attempt + 1}/{max_retries}): {str(e)}")
            print(f"Transaction: {tx}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            else:
                raise

    raise Exception(f"Failed to send transaction after {max_retries} attempts")


async def send_txn_from_csv(block_num, initial_nonce):
    df = pd.read_csv(f'data/txn/B_{block_num}.csv')
    tasks = []
    nonce = initial_nonce
    
    for index, row in df.iterrows():
        try:
            if pd.isna(row['to_address']) or row['to_address'] == '':
                recipient_address = sender_address
            else:
                recipient_address = Web3.to_checksum_address(str(row['to_address']))
            
            tx = {
                'to': recipient_address,
                'value': w3.to_wei(1, 'wei'),
                'from': sender_address,
                'gas': 21000,  # 标准交易的 gas 限制
                # 'gasPrice': await w3.eth.gas_price,
                'nonce': nonce
            }
            nonce += 1
            
            # print(f"准备发送交易: {tx}")
            
            task = asyncio.create_task(send_single_tx(tx))
            tasks.append(task)
        except Exception as e:
            print(f"创建交易时出错: {str(e)}")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    confirmation_tasks = []
    for result in results:
        if isinstance(result, Exception):
            print(f"交易发送失败: {str(result)}")
        else:
            tx_hash, tx_start_time = result
            # 检查 tx_hash 是否为有效的交易哈希
            if not isinstance(tx_hash, HexBytes):
                raise Exception(f"无效的交易哈希类型: {type(tx_hash)},tx_hash: {tx_hash}")
            
            confirmation_tasks.append(asyncio.create_task(confirm_transaction(tx_hash, tx_start_time, block_num)))

    return confirmation_tasks, nonce


async def confirm_transaction(tx_hash, tx_start_time, block_num):
    global total_transactions, transaction_times
    try:
        # Ensure tx_hash is HexBytes type
        if not isinstance(tx_hash, HexBytes):
            tx_hash = HexBytes(tx_hash)
        
        tx_receipt = await asyncio.wait_for(w3.eth.wait_for_transaction_receipt(tx_hash), timeout=300)  # 5 minutes timeout
        tx_end_time = time.time()
        
        confirmation_time = tx_end_time - tx_start_time
        total_transactions += 1
        transaction_times.append(confirmation_time)
        
        # print(f"Block {block_num}, Tx confirmed: {tx_hash.hex()}, Confirmation time: {confirmation_time:.4f} seconds")
    except asyncio.TimeoutError:
        print(f"Block {block_num}, Tx timeout: {tx_hash.hex()}, Not confirmed after 5 minutes")
    except Exception as e:
        print(f"Block {block_num}, Tx error: {tx_hash.hex()}, Error: {str(e)}")


async def handle_event(event):
    global w3, confirmation_tasks, blocks_processed, nonce
    # # get block number
    # block = await w3.eth.get_block(event)
    # print(f"Received block {block.number}, Time: {time.time()}")

    block_num = BASE_HEIGHT + blocks_processed
    print(f"Sending transactions for block {block_num}, Time: {time.time()}")
    
    tasks, nonce = await send_txn_from_csv(block_num, nonce)
    confirmation_tasks.extend(tasks)
    blocks_processed += 1
    
    print(f"Finished sending transactions for block {block_num}, Time: {time.time()}")
    if blocks_processed % 10 == 0:
        print_stats()


async def log_loop(event_filter, poll_interval):
    global start_time, blocks_processed, BLOCK_NUM, confirmation_tasks

    start_time = time.time()
    print(f"Start time: {start_time}")

    while blocks_processed < BLOCK_NUM:
        try:
            for event in await event_filter.get_new_entries():
                await handle_event(event)
            await asyncio.sleep(poll_interval)
        except Exception as e:
            print(f"Error processing new block: {str(e)}")
            await asyncio.sleep(1)  # Longer sleep on error

    print("Waiting for all transactions to be confirmed...")
    await asyncio.gather(*confirmation_tasks)


async def main():
    global w3, confirmation_tasks, blocks_processed, nonce

    # Get experiment name from command line arguments
    if len(sys.argv) < 2:
        print("Please provide an experiment name as a command line argument")
        sys.exit(1)
    experiment_name = sys.argv[1]

    w3 = await AsyncWeb3(AsyncWeb3.WebSocketProvider(LISTEN_URL))

    connect_status = await w3.is_connected()
    if not connect_status:
        raise Exception("Not connected to Ethereum node")
    
    nonce = await w3.eth.get_transaction_count(sender_address, 'pending')
    
    confirmation_tasks = []
    blocks_processed = 0

    block_filter = await w3.eth.filter('latest')
    await log_loop(block_filter, 0.1)

    # Uninstall block filter
    await w3.eth.uninstall_filter(block_filter.filter_id)

    print_stats(final=True, experiment_name=experiment_name)


def print_stats(final=False, experiment_name=None):
    global total_transactions, transaction_times, start_time
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if transaction_times:
        np_times = np.array(transaction_times)
        avg_latency = np.mean(np_times)
        median_latency = np.median(np_times)
        percentile_95 = np.percentile(np_times, 95)
        percentile_99 = np.percentile(np_times, 99)
    else:
        avg_latency = median_latency = percentile_95 = percentile_99 = 0
    
    throughput = total_transactions / elapsed_time
    
    print(f"{'Final' if final else 'Current'} Statistics:")
    print(f"Throughput: {throughput:.2f} transactions/second")
    print(f"Average transaction confirmation latency: {avg_latency:.4f} seconds")
    print(f"Median transaction confirmation latency: {median_latency:.4f} seconds")
    print(f"95th percentile transaction confirmation latency: {percentile_95:.4f} seconds")
    print(f"99th percentile transaction confirmation latency: {percentile_99:.4f} seconds")
    print(f"Total transactions: {total_transactions}")
    print(f"Total runtime: {elapsed_time:.2f} seconds")
    print("--------------------")

    # If final statistics, write results to file
    if final and experiment_name:
        os.makedirs("res", exist_ok=True)
        with open(f"res/{experiment_name}.txt", "w") as f:
            f.write(f"{experiment_name}\n")
            f.write(f"Throughput: {throughput:.2f} transactions/second\n")
            f.write(f"Average transaction confirmation latency: {avg_latency:.4f} seconds\n")


if __name__ == "__main__":
    asyncio.run(main())
