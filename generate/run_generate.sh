#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fp2_v2' \
    --audio_file 'audio/CS320/CS320.mp3' \
    --out_dir 'charts/CS320/d17' \
    --level 17 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'CS320' \
    --song_artist 'omg' \
    --sampling 'top-p' \
    ${@:1}
