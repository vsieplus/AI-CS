#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/aicsv1_1' \
    --audio_file 'audio/CS320/CS320.mp3' \
    --out_dir 'charts/CS320/s17' \
    --level 17 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'CS320' \
    --song_artist 'idk' \
    --sampling 'top-p' \
    ${@:1}
