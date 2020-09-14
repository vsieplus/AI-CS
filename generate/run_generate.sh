#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fiesta_p2_double' \
    --audio_file 'audio/asdlr/CS269.mp3' \
    --out_dir 'charts/asdlr/d19' \
    --level 19 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'CS269' \
    --song_artist 'omg' \
    --sampling 'top-p' \
    ${@:1}
