#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/prime12_all_single' \
    --audio_file 'audio/CS326/CS326.mp3' \
    --out_dir 'charts/CS326/S18' \
    --level 18 \
    --chart_format 'ucs' \
    --song_name 'errorcode' \
    --song_artist 'doin' \
    --sampling 'top-k' \
