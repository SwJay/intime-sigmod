import zmq
import asyncio
import time
import json
from zmq.asyncio import Context
from web3 import AsyncWeb3
import argparse
import sys
from collections import deque
import struct
import logging
import os

import numpy as np
from scipy.stats import lognorm
import compress
import outlier


class InTimeService:
    def __init__(self, node_index, node_num, host_ip, ip_list, ws_url, base_port):
        self.node_index = node_index
        self.node_num = node_num
        self.host_ip = host_ip
        self.ip_list = ip_list
        self.ws_url = ws_url
        self.base_port = base_port
        self.listen_port = base_port + node_index

        self.w3 = None
        self.context = Context()
        self.pub_socket = None
        self.sub_socket = None
        self.transaction_times = {}
        self.message_queue = deque(maxlen=100)  # Set maximum queue length
        self.batch_size = node_num * len(ip_list)  # Set batch processing size
        self.batch_timeout = 12  # Set batch processing timeout (seconds)
        # Set up logger
        self.logger = self.setup_logger(node_index)
        self.ar_file = f'ar_results.txt'

    def setup_logger(self, node_index):
        logger = logging.getLogger(f'InTimeService-{node_index}')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = 'intime_logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(f'{log_dir}/node_{node_index}.log')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Set log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    async def setup_zmq(self):
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{self.listen_port}")
        
        self.sub_socket = self.context.socket(zmq.SUB)
        for peer_ip in self.ip_list:
            for i in range(self.node_num):
                port = self.base_port + i
                self.sub_socket.connect(f"tcp://{peer_ip}:{port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')
        
        self.logger.info(f"Node {self.node_index} finished zmq setup")

    async def handle_new_transaction(self, tx):
        arrival_time = compress.timestamp_ms_encode(time.time())
        # self.logger.info(f"New transaction: {tx.hex()}, Arrival time: {arrival_time}")
        self.transaction_times[tx.hex()] = arrival_time

    async def handle_new_block(self, block):
        # print(f"New block: {block.hex()}, Arrival time: {time.time()}")
        full_block = await self.w3.eth.get_block(block, full_transactions=True)
        transactions = full_block.transactions
        
        self.logger.info(f"* Block {block.hex()} contains {len(transactions)} transactions")
        
        time_vector = []
        for tx in transactions:
            tx_hash = tx['hash'].hex()
            arrival_time = self.transaction_times.pop(tx_hash, compress.timestamp_ms_encode(time.time()))
            time_vector.append(arrival_time)
            # self.logger.info(f"** Transaction hash: {tx_hash}, Arrival time: {arrival_time}")
        
        if len(time_vector) > 0:
            self.logger.info(f"publish time_vector: {time_vector}")
            await self.publish_arrival_times(block.hex(), time_vector)

    async def publish_arrival_times(self, block_hash, time_vector):
        block_hash_bytes = bytes.fromhex(block_hash)
        compressed_vector = compress.compress_timestamps(time_vector)
        
        # Construct message: block_hash (32 bytes) + compressed_vector
        message = block_hash_bytes + compressed_vector
        
        await self.pub_socket.send(message)

    async def collect_time_vectors(self):
        is_judge_node = self.node_index == self.node_num - 1 and self.host_ip == JUDGE_IP
        
        while True:
            try:
                if is_judge_node:
                    await self.batch_process()
                else:
                    # Non-judge nodes only receive messages, no processing
                    await asyncio.wait_for(self.sub_socket.recv(), timeout=0.5)
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing received message: {e}")
                await asyncio.sleep(0.1)

    async def batch_process(self):
        batch_start_time = asyncio.get_event_loop().time()
        
        while len(self.message_queue) < self.batch_size:
            try:
                message = await asyncio.wait_for(self.sub_socket.recv(), timeout=0.1)
                self.message_queue.append(message)
            except asyncio.TimeoutError:
                pass
            
            if asyncio.get_event_loop().time() - batch_start_time > self.batch_timeout:
                break

        if self.message_queue:
            batch = [self.message_queue.popleft() for _ in range(len(self.message_queue))]
            await self.compute_ar(batch)

    async def compute_ar(self, batch):
        data = []
        block_hash = None
        for message in batch:
            try:
                # Parse binary message
                if block_hash is None:
                    block_hash = message[:32].hex()
                elif block_hash != message[:32].hex():
                    self.logger.warning(f"######## Block hash not identical: {block_hash} != {message[:32].hex()}")
                    continue

                compressed_vector = message[32:]
                
                self.logger.info(f"Received arrival time vector for block {block_hash}")
                
                decompressed_vector_ms = compress.decompress_timestamps(compressed_vector)
                decompressed_vector = [compress.timestamp_ms_decode(t) for t in decompressed_vector_ms]
                self.logger.info(f"decompressed_vector: {decompressed_vector}")

                # Generate 10 vectors based on this vector and add perturbation, add all to data
                for _ in range(10):
                    new_vector = decompressed_vector.copy()
                    for i in range(len(new_vector)):
                        noise = np.random.lognormal(mean=0, sigma=1) - 1 # E(noise) = 0
                        new_vector[i] += noise
                    data.append(new_vector)

                data.append(decompressed_vector)
            except Exception as e:
                self.logger.error(f"Unable to parse message: {e}")
        
        data = np.array(data)
        self.logger.info(f"Shape of merged data: {data.shape}")

        mus, sigmas, shifts = outlier.or_lmom_est_n(data)
        self.logger.info(f'mus: {mus}, sigmas: {sigmas}, shifts: {shifts}')
 
        n = data.shape[1]
        ars = np.zeros(n)
        
        block = await self.w3.eth.get_block(block_hash)
        end = block['timestamp']
        start = end - 12
        # self.logger.info(f"Block {block_hash}: {start} - {end}")

        for i in range(n):
            ars[i] = lognorm.cdf(end, sigmas[i], shifts[i], np.exp(mus[i])) - lognorm.cdf(start, sigmas[i], shifts[i], np.exp(mus[i]))
        
        # Write AR results to file
        with open(self.ar_file, 'a') as f:
            f.write(f'Block {block_hash}: AR = {ars}\n')

        self.logger.info('AR: %s', ars)

    async def log_loop(self, event_filter, event_type):
        while True:
            try:
                if event_type == 'pending':
                    for tx in await event_filter.get_new_entries():
                        await self.handle_new_transaction(tx)
                elif event_type == 'block':
                    for block in await event_filter.get_new_entries():
                        await self.handle_new_block(block)
            except Exception as e:
                self.logger.error(f"Error processing {event_type} event: {e}")
            await asyncio.sleep(0.1)

    async def run(self):
        await self.setup_zmq()
        self.w3 = await AsyncWeb3(AsyncWeb3.WebSocketProvider(self.ws_url))

        # check connection
        connect_status = await self.w3.is_connected()
        if not connect_status:
            raise Exception("Not connected to Ethereum node")

        pending_filter = await self.w3.eth.filter('pending')
        block_filter = await self.w3.eth.filter('latest')
    
        await asyncio.gather(
            self.log_loop(pending_filter, 'pending'),
            self.log_loop(block_filter, 'block'),
            self.collect_time_vectors()
        )

def parse_arguments():
    parser = argparse.ArgumentParser(description='InTime')
    parser.add_argument('-i', '--node_index', default=0, type=int, help='local node id')
    parser.add_argument('-n', '--node_num', default=1, type=int, help='local node num')
    parser.add_argument('-ip', '--host_ip', nargs='?', const='127.0.0.1', default='127.0.0.1', type=str, help='host ip')
    parser.add_argument('-f', '--ip_file', default='ip_list.json', type=str, help='ip list file')
    parser.add_argument('-ws', '--ws_port', default='8100', type=str, help='ws port')
    parser.add_argument('-p', '--base_port', default='9000', type=int, help='base port')
    parser.add_argument('-j', '--judge_ip', nargs='?', const='127.0.0.1', default='127.0.0.1', type=str, help='judge ip')
    args = parser.parse_args()
    
    if args.node_index < 0 or args.node_index >= args.node_num:
        parser.error(f"Invalid node index. Please use a number between 0 and {args.node_num - 1}.")
    
    return args


if __name__ == '__main__':
    print('InTime')
    args = parse_arguments()
    
    with open(args.ip_file, 'r') as f:
        ip_list = json.load(f)

    ws_url = f'ws://localhost:{args.ws_port}'

    global JUDGE_IP
    JUDGE_IP = args.judge_ip
    
    service = InTimeService(args.node_index, args.node_num, args.host_ip, ip_list, ws_url, args.base_port)
    asyncio.run(service.run())
