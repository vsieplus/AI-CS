#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/piuofficial_single' \
    --audio_file 'audio/bungee/bungee.mp3' \
    --out_dir 'charts/bungee/s16' \
    --level 16 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'bungee' \
    --song_artist 'OMG' \
    --sampling 'top-p' \
    ${@:1}
