#!/bin/bash

python generate.py \
    --model_dir '../train/models/double/fiesta_p2_double' \
    --audio_file 'audio/asdlr/CS269.mp3' \
    --out_dir 'charts/asdlr/d21' \
    --level 21 \
    --display_bpm 200 \
    --chart_format 'ucs' \
    --song_name 'asdlr' \
    --song_artist 'idk' \
    --sampling 'top-p' \
    ${@:1}
