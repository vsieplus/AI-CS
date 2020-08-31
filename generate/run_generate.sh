#!/bin/bash

python generate.py \
    --model_dir '../train/models/single/prime12_all_single' \
    --audio_file 'audio/asdlr/CS269.mp3' \
    --out_dir 'charts/asdlr/S22' \
    --level 22 \
    --chart_format 'ucs' \
    --song_name 'asdlr' \
    --song_artist 'idk' \
    --sampling 'beam-search' \