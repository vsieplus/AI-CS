#!/bin/bash
# Generate a dataset from json with the specified parameters

echo 'Building dataset...'
python3 processing/dataset_json.py \
    --splits='0.8,0.1,0.1' \
    --shuffle \
    --shuffle_seed=1949 \
    --song_types arcade remix fullsong shortcut \
    --min_difficulty=1 \
    --max_difficulty=28 \
    --permutations flip mirror flip_mirror \
    ${@:1}