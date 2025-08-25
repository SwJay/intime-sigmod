#!/bin/bash

set -exu
set -o pipefail

# INPUT
NUM_NODES=${NUM_NODES:-1}
ALL_NUM_NODES=${ALL_NUM_NODES:-1}
NAT=${NAT:-"none"}
BOOTNODE_ENODE=${BOOTNODE_ENODE:-}
PRYSM_BOOTSTRAP_NODE=${PRYSM_BOOTSTRAP_NODE:-}
JUDGE_HOST=${JUDGE_HOST:-"127.0.0.1"}
IF_INTIME=${IF_INTIME:-0}

HOST_IP=

# Set the path for the IP file
IP_FILE="intime/ip_list.json"
if [ ! -f "$IP_FILE" ]; then
    echo "Error: IP list file $IP_FILE does not exist"
    exit 1
fi

# if NAT!=none, then set NAT to extip:$NAT
if [ "$NAT" != "none" ]; then
  HOST_IP=$NAT
  NAT="extip:$NAT"
fi

DEVNET_DIR="./devnet"
CHAIN_ID=32382

GETH_BOOTNODE_PORT=30301

GETH_HTTP_PORT_BASE=8000
GETH_WS_PORT_BASE=8100
GETH_AUTH_RPC_PORT_BASE=8200
GETH_METRICS_PORT_BASE=8300
GETH_P2P_PORT_BASE=8400

PRYSM_BEACON_RPC_PORT_BASE=4000
PRYSM_BEACON_GRPC_GATEWAY_PORT_BASE=4100
PRYSM_BEACON_P2P_TCP_PORT_BASE=4200
PRYSM_BEACON_P2P_UDP_PORT_BASE=4300
PRYSM_BEACON_MONITORING_PORT_BASE=4400

PRYSM_VALIDATOR_RPC_PORT_BASE=7000
PRYSM_VALIDATOR_GRPC_GATEWAY_PORT_BASE=7100
PRYSM_VALIDATOR_MONITORING_PORT_BASE=7200

# clear old data
if [ -n "$(docker ps -aq)" ]; then
  docker rm -f $(docker ps -a -q)
fi
rm -rf "$DEVNET_DIR"
mkdir -p "$DEVNET_DIR"
chmod 777 "$DEVNET_DIR"
cp ./config.yml $DEVNET_DIR/
cp ./genesis.json $DEVNET_DIR/
cp -r ./keystore $DEVNET_DIR/
# intime
rm -rf intime_logs
mkdir -p intime_logs
rm -f ar_results.txt

if [[ -z "${BOOTNODE_ENODE}" ]]; then
  # start bootnode
  docker run --rm -v $DEVNET_DIR:/bootnode ethereum/client-go:alltools-v1.14.11 bootnode -genkey /bootnode/nodekey
  docker run -d --name bootnode -v $DEVNET_DIR:/bootnode -p $GETH_BOOTNODE_PORT:$GETH_BOOTNODE_PORT/udp ethereum/client-go:alltools-v1.14.11 bootnode -nodekey /bootnode/nodekey -nat $NAT -addr 0.0.0.0:$GETH_BOOTNODE_PORT -verbosity 5

  sleep 5

  # get bootnode's enode ID
  BOOTNODE_ID=$(docker logs bootnode 2>&1 | grep "enode://" | tail -n 1 | awk -F'enode://' '{print $2}' | awk -F'@' '{print $1}')

  if [ -z "$HOST_IP" ]; then
    BOOTNODE_ENODE="enode://${BOOTNODE_ID}@$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' bootnode):$GETH_BOOTNODE_PORT"
  else
    BOOTNODE_ENODE="enode://${BOOTNODE_ID}@$HOST_IP:$GETH_BOOTNODE_PORT"
  fi
  echo "MYVAR: BOOTNODE_ENODE=$BOOTNODE_ENODE"

# generate genesis state
  docker run --rm -v $DEVNET_DIR:/data gcr.io/prysmaticlabs/prysm/cmd/prysmctl:latest testnet generate-genesis \
    --fork=capella \
    --num-validators=$NUM_NODES \
    --chain-config-file=/data/config.yml \
    --geth-genesis-json-in=/data/genesis.json \
    --output-ssz=/data/genesis.ssz \
    --geth-genesis-json-out=/data/genesis.json \

else
  cp ./genesis.ssz $DEVNET_DIR/
fi


for i in $(seq 0 $((NUM_NODES-1))); do
  NODE_DIR="$DEVNET_DIR/node-$i"
  mkdir -p "$NODE_DIR/execution" "$NODE_DIR/consensus"
  cp "$DEVNET_DIR/genesis.json" "$NODE_DIR/execution/"
  cp "$DEVNET_DIR/genesis.ssz" "$NODE_DIR/consensus/"
  cp "$DEVNET_DIR/config.yml" "$NODE_DIR/consensus/"
  cp -r "$DEVNET_DIR/keystore" "$NODE_DIR/execution"
  echo "" > "$NODE_DIR/execution/geth_password.txt"
  
  GETH_ACCOUNT=$(docker run --rm -v $NODE_DIR/execution:/execution ethereum/client-go:v1.14.11 account new --datadir /execution --password /execution/geth_password.txt | awk -F'{|}' '{print $2}')
  
  # create JWT secret
  openssl rand -hex 32 | tr -d "\n" > "$NODE_DIR/execution/jwtsecret"

  # start node
  GETH_HTTP_PORT=$((GETH_HTTP_PORT_BASE + i))
  GETH_WS_PORT=$((GETH_WS_PORT_BASE + i))
  GETH_AUTH_RPC_PORT=$((GETH_AUTH_RPC_PORT_BASE + i))
  GETH_METRICS_PORT=$((GETH_METRICS_PORT_BASE + i))
  GETH_P2P_PORT=$((GETH_P2P_PORT_BASE + i))

  PRYSM_BEACON_RPC_PORT=$((PRYSM_BEACON_RPC_PORT_BASE + i))
  PRYSM_BEACON_GRPC_GATEWAY_PORT=$((PRYSM_BEACON_GRPC_GATEWAY_PORT_BASE + i))
  PRYSM_BEACON_P2P_TCP_PORT=$((PRYSM_BEACON_P2P_TCP_PORT_BASE + i))
  PRYSM_BEACON_P2P_UDP_PORT=$((PRYSM_BEACON_P2P_UDP_PORT_BASE + i))
  PRYSM_BEACON_MONITORING_PORT=$((PRYSM_BEACON_MONITORING_PORT_BASE + i))

  PRYSM_VALIDATOR_RPC_PORT=$((PRYSM_VALIDATOR_RPC_PORT_BASE + i))
  PRYSM_VALIDATOR_GRPC_GATEWAY_PORT=$((PRYSM_VALIDATOR_GRPC_GATEWAY_PORT_BASE + i))
  PRYSM_VALIDATOR_MONITORING_PORT=$((PRYSM_VALIDATOR_MONITORING_PORT_BASE + i))
  
  NODE_DIR=$NODE_DIR \
  GETH_ACCOUNT=$GETH_ACCOUNT \
  BOOTNODE_ENODE=$BOOTNODE_ENODE \
  CHAIN_ID=$CHAIN_ID \
  GETH_HTTP_PORT=$GETH_HTTP_PORT \
  GETH_WS_PORT=$GETH_WS_PORT \
  GETH_AUTH_RPC_PORT=$GETH_AUTH_RPC_PORT \
  GETH_METRICS_PORT=$GETH_METRICS_PORT \
  GETH_P2P_PORT=$GETH_P2P_PORT \
  PRYSM_BEACON_RPC_PORT=$PRYSM_BEACON_RPC_PORT \
  PRYSM_BEACON_GRPC_GATEWAY_PORT=$PRYSM_BEACON_GRPC_GATEWAY_PORT \
  PRYSM_BEACON_P2P_TCP_PORT=$PRYSM_BEACON_P2P_TCP_PORT \
  PRYSM_BEACON_P2P_UDP_PORT=$PRYSM_BEACON_P2P_UDP_PORT \
  PRYSM_BEACON_MONITORING_PORT=$PRYSM_BEACON_MONITORING_PORT \
  PRYSM_VALIDATOR_RPC_PORT=$PRYSM_VALIDATOR_RPC_PORT \
  PRYSM_VALIDATOR_GRPC_GATEWAY_PORT=$PRYSM_VALIDATOR_GRPC_GATEWAY_PORT \
  PRYSM_VALIDATOR_MONITORING_PORT=$PRYSM_VALIDATOR_MONITORING_PORT \
  PRYSM_BOOTSTRAP_NODE=$PRYSM_BOOTSTRAP_NODE \
  HOST_IP=$HOST_IP \
  NAT=$NAT \
  MIN_SYNC_PEERS=$((ALL_NUM_NODES/2)) \
  VALIDATOR_INDEX=$i \
  NODE_INDEX=$i \
  docker-compose -f docker-compose.yml -p node-$i up -d

  # Check if the PRYSM_BOOTSTRAP_NODE variable is already set
  if [[ -z "${PRYSM_BOOTSTRAP_NODE}" ]]; then
      sleep 5 # sleep to let the prysm node set up
      # If PRYSM_BOOTSTRAP_NODE is not set, execute the command and capture the result into the variable
      # This allows subsequent nodes to discover the first node, treating it as the bootnode
      PRYSM_BOOTSTRAP_NODE=$(curl -s localhost:$PRYSM_BEACON_GRPC_GATEWAY_PORT_BASE/eth/v1/node/identity | jq -r '.data.enr')
          # Check if the result starts with enr
      if [[ $PRYSM_BOOTSTRAP_NODE == enr* ]]; then
          echo "MYVAR: PRYSM_BOOTSTRAP_NODE=$PRYSM_BOOTSTRAP_NODE"
      else
          echo "PRYSM_BOOTSTRAP_NODE does NOT start with enr"
          exit 1
      fi
  fi

  # intime
  if [ "${IF_INTIME}" -eq 1 ]; then
    echo "IF_INTIME: ${IF_INTIME}"
    /opt/conda/envs/web3/bin/python3 intime/intime.py -i $i -n $NUM_NODES -ip $HOST_IP -f $IP_FILE -ws $((GETH_WS_PORT_BASE + i)) -j $JUDGE_HOST > /dev/null 2>&1 &
  fi

done

echo "Devnet deployed, $NUM_NODES nodes"