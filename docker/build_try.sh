set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

docker build -t seed_rl:dmlab_0.0 -f docker/Dockerfile.dmlab1 .
# docker run --gpus all  -v ~/:/outdata -itd seed_rl:dmlab bash
docker build -t seed_rl:dmlab_0.1 -f docker/Dockerfile.dmlab2 .