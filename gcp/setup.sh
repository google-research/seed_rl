#!/bin/bash
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e
PROJECT_ID=$(gcloud config get-value project)
export IMAGE_URI=gcr.io/$PROJECT_ID/seed

start_training () {
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  $DIR/../docker/build.sh
  $DIR/../docker/push.sh
  # Create bucket if doesn't exist.
  gsutil ls gs://seed_rl || gsutil mb gs://seed_rl
  JOB_NAME="SEED_$(date +"%Y%m%d%H%M%S")"
  # Start training on AI platform.
  gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --project=${PROJECT_ID} \
    --job-dir gs://seed_rl/${JOB_NAME} \
    --region us-central1 \
    --config /tmp/config.yaml \
    --stream-logs -- --environment=${ENVIRONMENT} --agent=${AGENT} \
    --actors_per_worker=${ACTORS_PER_WORKER} --workers=${WORKERS} --
}
