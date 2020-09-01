#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/prime12_all_single' \
    --audio_file 'audio/errorcode/errorcode0.mp3' \
    --out_dir 'charts/errorcode/s18' \
    --level 18 \
    --display_bpm 200 \
    --chart_format 'ucs' \
    --song_name 'bungee' \
    --song_artist 'OMG' \
    --sampling 'top-k' \
    ${@:1}