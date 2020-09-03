#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fiesta_p2_double' \
    --audio_file 'audio/bungee/bungee.mp3' \
    --out_dir 'charts/bungee/d14' \
    --level 14 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'bungee' \
    --song_artist 'OMG' \
    --sampling 'top-k' \
    ${@:1}
