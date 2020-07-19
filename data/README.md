# data

This directory contains code for retrieving and processing the data
to be used to train the models.

## Getting the data

To download the raw data call `./get_data.sh TYPE`, where `TYPE` is one of
`base`, `ucs`, or `all`. `base` downloads the base packs as included in
STEPF2/P1, while `ucs` downloads charts from the UCS website. `all` does both. Calling the scripts in `processing` will then generate the appropriate json files for the charts.

```
./get_data.sh all > dataset/raw 
./processing/generate_json.py > dataset/json
./processing/process_json.py > dataset/json
```


## Dataset info

The majority of the chart data used for this project 
is taken from the collection provided by the [STEPF2/P1 Project](https://stepf2.blogspot.com/).
Specifically, we include packs starting from 1st-3rd up to Prime 2,
including Pro 1 & 2, as well as Infinity 1.10. When training the
models, it is possible to experiment with different subsets of the
data (i.e. train only on charts that appeared in a certain mix, or charts from songs
categorized under a certain genre, etc.).

There is also functionality to train on UCS chart data, downloaded from
the official PIU UCS sharing [site](http://www.piugame.com/bbs/board.php?bo_table=ucs). You may also 
add your own `.ucs` files. As above, there is functionality for training 
the models on a particular collection of UCS.

Each type of chart data is associated with some basic information, such 
as the song itself, artist, chart type, level, bpm, and the actual step chart. This
information is used to help train the models to learn how to generate charts.