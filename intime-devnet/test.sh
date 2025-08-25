#!/bin/bash

set -exu
set -o pipefail

IF_INTIME_VALUES=(0 1)
NUM_NODES_VALUES=(1 2 3 4 5)

# OPERATION="git_pull" bash ./start_remote.sh

for num_nodes in "${NUM_NODES_VALUES[@]}"; do
  for if_intime in "${IF_INTIME_VALUES[@]}"; do  
    echo "Start test for IF_INTIME=$if_intime, NUM_NODES=$num_nodes"
    
    IF_INTIME=$if_intime NUM_NODES=$num_nodes OPERATION="run_devnet" bash ./start_remote.sh

    sleep 10
    
    python send_txn.py ${num_nodes}-${if_intime}

    OPERATION="clean" bash ./start_remote.sh
    
    echo "Complete test for IF_INTIME=$if_intime, NUM_NODES=$num_nodes"
    echo "-----------------------------------"
  done
done

echo "All done"