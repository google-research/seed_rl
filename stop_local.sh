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



get_descendants ()
{
  local children=$(ps -o pid= --ppid "$1")

  for pid in $children
  do
    get_descendants "$pid"
  done
  if (( $1 != $$ && $1 != $PPID )); then
    echo "$1 "
  fi
}

processes=''
for C in `tmux list-panes -s -F "#{pane_pid} #{pane_current_command}" 2> /dev/null | grep -v tmux | awk '{print $1}'`; do
  processes+=$(get_descendants $C)
done
if [[ $processes != '' ]]; then
  kill -9 $processes
fi
tmux kill-session -t seed_rl
