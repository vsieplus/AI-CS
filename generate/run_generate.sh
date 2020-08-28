#!/bin/bash

python generate.py \
    --model_dir '../train/models/prime12_all_single' \
    --audio_file 'audio/CS326/CS326.mp3' \
    --out_dir 'charts/CS326/S17' \
    --level 17 \
    --chart_format 'ucs' \
    --song_name 'cs326' \
    --song_artist 'idk' \