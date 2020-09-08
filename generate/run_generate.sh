#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/prime12_all_double' \
    --audio_file 'audio/CS319/CS319.mp3' \
    --out_dir 'charts/district1/d17' \
    --level 17 \
    --display_bpm 150 \
    --chart_format 'both' \
    --song_name 'district1' \
    --song_artist 'max' \
    --sampling 'top-p' \
    ${@:1}
