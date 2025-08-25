#!/bin/bash

set -exu
set -o pipefail


BOOT_HOST="54.238.7.205"
REMOTE_HOSTS=("3.112.43.41")
# REMOTE_HOSTS=(
#   "54.250.151.148" \
#   "52.69.239.133" \
#   "35.77.94.213" \
#   "3.82.37.119" \
#   "54.211.157.122" \
#   "34.235.88.110" \
#   "54.88.184.246" \
#   "3.252.145.6" \
#   "34.249.21.96" \
#   "54.154.158.29" \
#   "34.253.130.232"
#   )
JUDGE_HOST="34.253.130.232"

# OPERATION="clean"
# OPERATION="git_pull"
OPERATION=${OPERATION:-"run_devnet"}

REMOTE_SCRIPT="./run_devnet.sh"

IF_INTIME=${IF_INTIME:-0}
NUM_NODES=${NUM_NODES:-1}
ALL_NUM_NODES=${ALL_NUM_NODES:-1}
NAT=$BOOT_HOST
BOOTNODE_ENODE=
PRYSM_BOOTSTRAP_NODE=

SSH_OPTIONS="-o StrictHostKeyChecking=no"

REMOTE_USER="ubuntu"
KEY_PATH="~/.ssh/us-east-1.pem"
KEY_LOCAL_PATH="$HOME/.ssh/us-east-1.pem"


if [ "$OPERATION" = "clean" ]; then
  REMOTE_SCRIPT="./clean.sh"
  ssh -i $KEY_PATH $REMOTE_USER@$BOOT_HOST "cd ~/intime-sigmod/intime-devnet && sudo bash $REMOTE_SCRIPT"
  if [ ${#REMOTE_HOSTS[@]} -gt 0 ]; then
    for host in "${REMOTE_HOSTS[@]}"; do
      ssh -i $KEY_PATH $REMOTE_USER@$host "cd ~/intime-sigmod/intime-devnet && sudo bash $REMOTE_SCRIPT" &
    done
    wait
  fi
  bash ./clean.sh
elif [ "$OPERATION" = "git_pull" ]; then
  ssh -i $KEY_PATH $SSH_OPTIONS $REMOTE_USER@$BOOT_HOST "cd ~/intime-sigmod/intime-devnet && git pull"
  if [ ${#REMOTE_HOSTS[@]} -gt 0 ]; then
    for host in "${REMOTE_HOSTS[@]}"; do
      ssh -i $KEY_PATH $SSH_OPTIONS $REMOTE_USER@$host "cd ~/intime-sigmod/intime-devnet && git pull" &
    done
    wait
  fi
else
  # run devnet on bootnode, get enode and enr
  output=$(ssh -i $KEY_PATH $REMOTE_USER@$BOOT_HOST \
    "cd ~/intime-sigmod/intime-devnet && \
    sudo NUM_NODES=$NUM_NODES \
    ALL_NUM_NODES=$ALL_NUM_NODES \
    NAT=$NAT \
    BOOTNODE_ENODE=$BOOTNODE_ENODE \
    PRYSM_BOOTSTRAP_NODE=$PRYSM_BOOTSTRAP_NODE \
    JUDGE_HOST=$JUDGE_HOST \
    IF_INTIME=$IF_INTIME \
    bash $REMOTE_SCRIPT")
  # 解析输出并设置本地变量
  vars=$(echo "$output" | grep "^MYVAR: " | sed 's/^MYVAR: //')
  eval "$vars"
  # echo "result NUM_NODES: $NUM_NODES"
  echo "result BOOTNODE_ENODE: $BOOTNODE_ENODE"
  echo "result PRYSM_BOOTSTRAP_NODE: $PRYSM_BOOTSTRAP_NODE"

  # get generated genesis files: genesis.json, genesis.ssz
  ssh -i $KEY_PATH $REMOTE_USER@$BOOT_HOST "sudo chmod 644 ~/intime-sigmod/intime-devnet/devnet/genesis.json ~/intime-sigmod/intime-devnet/devnet/genesis.ssz"
  scp -i $KEY_PATH $KEY_LOCAL_PATH $REMOTE_USER@$BOOT_HOST:~/.ssh/

  if [ ${#REMOTE_HOSTS[@]} -gt 0 ]; then
    for host in "${REMOTE_HOSTS[@]}"; do
      ssh -i $KEY_PATH $REMOTE_USER@$BOOT_HOST << EOF &
      scp -i $KEY_PATH $SSH_OPTIONS ~/intime-sigmod/intime-devnet/devnet/genesis.json $REMOTE_USER@$host:~/intime-sigmod/intime-devnet/
      scp -i $KEY_PATH ~/intime-sigmod/intime-devnet/devnet/genesis.ssz $REMOTE_USER@$host:~/intime-sigmod/intime-devnet/
      ssh -i $KEY_PATH $REMOTE_USER@$host \
        "cd ~/intime-sigmod/intime-devnet && \
        sudo NUM_NODES=$NUM_NODES \
        ALL_NUM_NODES=$ALL_NUM_NODES \
        NAT=$host \
        BOOTNODE_ENODE='$BOOTNODE_ENODE' \
        PRYSM_BOOTSTRAP_NODE='$PRYSM_BOOTSTRAP_NODE' \
        JUDGE_HOST=$JUDGE_HOST \
        IF_INTIME=$IF_INTIME \
        bash $REMOTE_SCRIPT"
EOF
    done
    wait
  fi
  
  # local
  # scp -i $KEY_PATH $REMOTE_USER@$BOOT_HOST:~/intime-sigmod/intime-devnet/devnet/genesis.json ./
  # scp -i $KEY_PATH $REMOTE_USER@$BOOT_HOST:~/intime-sigmod/intime-devnet/devnet/genesis.ssz ./
  # NUM_NODES=1 ALL_NUM_NODES=$ALL_NUM_NODES BOOTNODE_ENODE=$BOOTNODE_ENODE PRYSM_BOOTSTRAP_NODE=$PRYSM_BOOTSTRAP_NODE IF_INTIME=0 bash ./run_devnet.sh
fi