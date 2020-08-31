#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/prime12_all_single' \
    --audio_file 'audio/asdlr/CS269.mp3' \
    --out_dir 'charts/asdlr/S21' \
    --level 21 \
    --chart_format 'ucs' \
    --song_name 'asdlr' \
    --song_artist 'idk' \
    --sampling 'top-k' \
    ${@:1}