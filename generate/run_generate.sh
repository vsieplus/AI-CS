#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/fiesta_p2_single' \
    --audio_file 'audio/bungee/bungee.mp3' \
    --out_dir 'charts/bungee/s19' \
    --level 19 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'bungee' \
    --song_artist 'OMG' \
    --sampling 'top-k' \
    ${@:1}
