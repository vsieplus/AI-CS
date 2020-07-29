# data

This directory contains code for retrieving and processing the data
to be used to train the models.

| Directory          | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `dataset/raw/`     | Raw ucs/scc packs                                             |
| `dataset/json/`    | Chart data in json format                                     |
| `dataset/subsets/` | Metadata representing dataset subsets                         |
| `processing/`      | ssc/ucs file extraction, conversion to json, dataset creation |
| `spiders/`         | Web scrapers for the UCS site |

## Dataset info

The majority of the chart data used for this project is taken from the collection
provided by the [STEPF2/P1 Project](https://stepf2.blogspot.com/). Specifically, it includes
packs starting from 1st-3rd up to Prime 2, including Pro 1 & 2, as well as Infinity 1.10.
When training a model, you can create a sub-dataset as shown below to customize what
types of songs and charts it sees. This may directly affect how it behaves afterwards,
when it tries to generate something brand new.

There is also functionality to train on UCS chart data, downloaded from
the official PIU UCS sharing [site](http://www.piugame.com/bbs/board.php?bo_table=ucs).

Each step chart is associated with some basic information, such as a path to the
song itself, the artist and step artist (sometimes), chart type, level, bpm,
and the actual step chart. This information is used to help train the models to
learn how to generate charts. The final representation of the chart in the json
files amounts to effectively a 3-tuple for each beat.

## Getting the data

To download the 'official' game chart data you may call `./get_stepp1_data.sh`, which
downloads the base packs as included in STEPF2/P1. These files will be stored under
`dataset/raw/`, similar to the pack data layout in stepf2/p1. Alternatively, if
you already have the packs on your own computer, you may simply copy them over
(treat `dataset/raw` as the equivalent of the `Songs/` directory).

To download UCS from the official PIU UCS site, you can run `ucs_scrape.py` with
different options. For instance, calling the below will download all UCS singles charts
created by step artists **artist1** or **artist2**, for songs that debuted in either
prime 2 or fiesta, and are between level 14 and 16. It is possible to specify
other parameters such as specific songs, chart types, date published, etc...

```python
python scrape_ucs.py \
    --chart_type=single \
    --step_artists artist1 artist2 \
    --min_level=14 \
    --max_level=16
```

This will create a custom ucs 'pack', (which it will ask you to name) under
`dataset/raw/UCS_packname`.

You may also add your own `.ucs` files. However, if it is a plain `.ucs` file as
created in StepEdit lite, you will need to enter some additional metadata about the charts.
To do so, first gather the ucs files (in subfolders) into some directory
`UCS_PACKNAME` and copy it under `dataset/raw/`. If the ucs is originally from the
official site, you may skip copying the audio if you wish. If it is a non-official
ucs file, you should include the audio file. Then call

`python ucs_add_metadata.py --ucs_dir=dataset/raw/UCS_PACKNAME`

which will walk you through the process. For ucs files originally from the UCS
sharing site, with the same code name (e.g. 'CS280') some data can be automatically
found. Either way, some additional information will be needed like chart type,
level, etc. For the scraped files, this metadata is gathered automatically during
the scraping process.

Note that the first time scraping/working with ucs files, it will first scrape
some general metadata and download the audio and templates from the UCS site. In
addition, ucs files with metadata cannot be guaranteed to work properly with other
programs like stepedit lite or some UCS viewers.

## Processing the data

To convert 'ssc/ucs' files to usable format, there are several scripts in `processing/`.
First to extract data from the ssc/ucs files and convert to json, you can call

`python processing/generate_json.py`

The resulting json files will be located under `dataset/json` according to their
respective packs.

To specify a custom subset of charts you can create a dataset. Calling
`bash build_dataset.sh` will call `processing/dataset_json.py` with the specified
parameters, ranging from difficulty levels, chart type (single/double),
song type (arcade/remix/...) and more. You may customize these parameters to build
your own datasets. This script will also prompt you for a dataset **name** and
ask which packs to choose songs and charts from.

`bash build_dataset.sh [--additional --parameters ...]`

The resulting files will be stored in `dataset/subsets/name` and will act as metadata
to be used when training is initiated for a particular model (see `train/`).
