#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/prime12_all_single' \
    --audio_file 'audio/bungee/bungee.mp3' \
    --out_dir 'charts/bungee/S15' \
    --level 15 \
    --chart_format 'ucs' \
    --song_name 'bungee' \
    --song_artist 'OMG' \
    --sampling 'top-p' \
    ${@:1}