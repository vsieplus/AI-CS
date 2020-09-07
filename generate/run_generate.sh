#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fiesta_p2_double' \
    --audio_file 'audio/CS319/CS319.mp3' \
    --out_dir 'charts/district1/d15' \
    --level 15 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'district1' \
    --song_artist 'max' \
    --sampling 'top-k' \
    ${@:1}
