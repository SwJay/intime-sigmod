# Readme

This the code repository for paper ID 567.


## Intime

### Data

- `intime/data/blocks.csv` contains the Ethereum blocks from blockNumber 19000000 to 19000100.

- `intime/data/receipts.csv` contains the 18,484 transaction receipts correspondingly.

Both real datasets are extracted using [Ethereum ETL](https://github.com/blockchain-etl/ethereum-etl).

### Requirements

- Running environment: Centos 7, Python 3.8

- Requirements: `intime/requirements.txt`

### Code

- `intime/main.py` contains all test cases for experiments.

- `intime/agnostic.py` contains algorithms for agnostic mean estimation.

- `intime/outlier.py` contains algorithms for outlier removal, parameter estimation.

- `intime/receipt.py` contains codes for real data (`data/blocks.csv`, `data/receipts.csv`) processing.

- `intime/draw.py` contains codes for experimental figure ploting.

### Instructions

After installing required packages, directly run `python main.py` to execute all exeperiments.


## Intime-devnet

We set up a private PoS Ethereum devnet among geo-distributed nodes. We use [Geth](https://github.com/ethereum/go-ethereum) as the Ethereum client, and [Prysm](https://github.com/prysmaticlabs/prysm) as the consensus client. 

### Data

- `intime-devnet/data/txn` contains 18,484 transactions in Ethereum blocks from blockNumber 19000000 to 19000100.

### Requirements

- Running environment: Ubuntu 24.04 LTS, Python 3.9, Docker 27.3.1

- Requirements: `intime-devnet/requirements.txt`

### Config

- `intime-devnet/config.yaml` contains the configuration for Prysm.

- `intime-devnet/genesis.json` is the genesis file for the Ethereum chain.

- `intime-devnet/docker-compose.yaml` is the docker compose file for running a single node with Prysm and Geth.

### Code

- `intime-devnet/run_devnet.sh` is the script to run the devnet on a local machine.

- `intime-devnet/start_remote.sh` is the script to run a global devnet among multiple physical machines, it will set up a bootnode first and then start the rest nodes to form the devnet.

- `intime-devnet/clean.sh` is the script to clean the devnet on a local machine.

- `intime-devnet/intime/intime.py` runs the intime instance that (1) records the arrival tiem for listened transactions; (2) when receiving a new block, broadcast the corresponding time vector to other nodes; (3) the judge node collects the time vectors from all nodes and estimates the arrival time for each transaction.

- `intime-devnet/intime/compress.py` contains the code for the compression algorithm and experiments.

- `intime-devnet/intime/attest_size.py` contains the code to fetch real attestations and compute sizes.

- `intime-devnet/intime/send_txn.py` sends transactions from the real dataset to the devnet and records the throughput and latency.

### Instructions

1. Setup servers and clone the repo to every server.
    - IPs of servers should be stored in `intime-devnet/ip_list.json`, and correspondingly set the BOOT_HOST, REMOTE_HOSTS, JUDGE_HOST in `intime-devnet/start_remote.sh`.
    - SSH settings like user name and path to ssh keys connecting other servers (REMOTE_USER, KEY_PATH, KEY_LOCAL_PATH) should also be updated in `intime-devnet/start_remote.sh`.

2. Run `IF_INTIME=<if_intime> ALL_NUM_NODES=<all_num_nodes> OPERATION="run_devnet" bash start_remote.sh` to start the devnet, where

    - `if_intime` is a boolean value indicating whether to run intime on top of the devnet;

    - `all_num_nodes` is the total number of nodes in the devnet.

3. Run `python send_txn.py <all_num_nodes>-<if_intime>` to send transactions to the devnet and record the throughput and latency.
