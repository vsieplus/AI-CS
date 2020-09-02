#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fiesta_p2_double' \
    --audio_file 'audio/errorcode/errorcode0.mp3' \
    --out_dir 'charts/errorcode/d25' \
    --level 25 \
    --display_bpm 200 \
    --chart_format 'ucs' \
    --song_name 'errorcode' \
    --song_artist 'doin' \
    --sampling 'top-k' \
    ${@:1}
