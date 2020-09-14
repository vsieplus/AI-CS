#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fp2_exhh' \
    --audio_file 'audio/asdlr/CS269.mp3' \
    --out_dir 'charts/asdlr/d16' \
    --level 16 \
    --display_bpm 150 \
    --chart_format 'ucs' \
    --song_name 'CS269' \
    --song_artist 'idk' \
    --sampling 'top-p' \
    ${@:1}
