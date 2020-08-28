#!/bin/bash

python generate.py \
    --model_dir '../train/models/prime12_all_single' \
    --audio_file 'audio/errorcode/errorcode0.mp3' \
    --out_dir 'charts/errorcode/S18' \
    --level 18 \
    --chart_format 'ucs' \
    --song_name 'errorcode' \
    --song_artist 'doin' \
    --sampling 'top-k' \
    -k 25