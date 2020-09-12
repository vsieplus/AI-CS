#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/p12_exh' \
    --audio_file 'audio/CS294/CS294.mp3' \
    --out_dir 'charts/CS294/s17' \
    --level 17 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'CS294' \
    --song_artist 'omg' \
    --sampling 'top-k' \
    ${@:1}
