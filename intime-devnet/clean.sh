pkill -f intime.py
wait $(pgrep -f intime.py) 2>/dev/null || true

if [ -n "$(docker ps -aq)" ]; then
  docker rm -f $(docker ps -a -q)
fi
rm -rf ./devnet
rm -rf ./genesis.ssz