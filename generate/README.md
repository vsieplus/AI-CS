# generate

This directory contains code used to generate a stepchart, given a set of already 
trained models from `../train`. For convenience, it can help to create subdirectories
in this folder, one for music files, and one where your charts will be saved to.

### Example

The command below uses the model files in `model_dir` to generate a step chart
with the specified options. The level can be from 1-28, while you can choose the 
`chart_format` to be 'ucs', 'ssc', or 'both'. The output files will be saved to the
specified `out_dir`, along with a copy of the specified `audio_file`.

```python
python generate.py \
    --model_dir '../train/models/testmodel' \
    --audio_file 'audio/iolite_sky.mp3' \
    --out_dir 'charts/Iolite_sky/S16' \
    --level 16 \
    --chart_format 'ucs' \
    --song_name 'Iolite Sky' \
    --song_artist 'Doin' \
```

Other generation options can be specified (see the script and `run_generate.sh` for more info).

Note that `--bpm` is not required, and is simply used to generate the measure/beat/split
subdivisions in the actual chart format. It will also affect the way the chart is viewed
when auto velocity is not enabled, but the timings of the steps are the same regardless of the
bpm. If left blank, it will default to 140.