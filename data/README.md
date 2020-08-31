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

This project is designed to handle Pump it Up step chart data contained in both `.ssc` and `.ucs` files.
SSC files are created by stepmakers to work with certain versions of stepmania; for PIU in particular,
most commonly StepF2/P1. Packs are often shared on YouTube, Facebook, and on other online forums. UCS files
can be created with [StepEdit Lite](http://www.piugame.com/piu.ucs/ucs.intro/ucs.intro.php), a program provided by Andamiro which players can use to write custom step charts. Many UCS charts are shared [online](http://www.piugame.com/bbs/board.php?bo_table=ucs), where others may view, download, and play other players' charts.

Each step chart is associated with some basic information, such as its corresponding song, the step artist (sometimes),
its type (single / double), difficulty level, bpm, and the sequence of steps itself. Together, the audio and the
sequence of steps forms a series of correspondences between audio features and steps, which is what we end up passing
to the models. We can also use other chart features as a secondary form of data, such as the chart level and type.

## Getting the data

Step chart files should be organized into pack folders, and placed under `dataset/raw` (treat it as the
equivalent of the `Songs/` directory if you're familiar with StepMania/StepF2).

SSC packs can be found on various online forums created by different step artists. 

To download multiple UCS files at once from the official PIU UCS site, you can run `ucs_scrape.py` with
different options. For instance, calling the below will download all UCS singles charts
created by step artists **artist1** or **artist2**, and are between level 14 and 16. 
It is possible to specify other parameters such as specific song titles and a range for the date
published. Use the `-h` option for more info. This script will prompt you for your PIU account
login information, as an account is required to download UCS files from their website.

Note that the first time scraping/working with ucs files, you should first run 

```ucs_scrape_meta.py --scrape_meta --download```

to first scrape some general metadata and download the audio and templates from the UCS site. This will
create a placeholder pack with chart under `dataset/raw/00-UCS-BASE`. These files will be referred to for downloaded ucs charts in order to avoid duplication of audio files.

```python
python scrape_ucs.py \
    --pack_name='pack_name'
    --chart_type=single \
    --step_artists artist1 artist2 \
    --min_level=14 \
    --max_level=16
```

This will create a custom ucs 'pack' under `dataset/raw/pack_name`.

You may also add your own `.ucs` files. However, if it is a plain `.ucs` file as
created in StepEdit lite, you will need to enter some additional metadata about the charts.
To do so, first gather the ucs files (in subfolders with audio/ucs) into some directory
`ucs_packname` and copy it under `dataset/raw/`. If the ucs is originally from the official site,
you may skip copying the audio if you wish. If it is a non-official ucs file, you should include the 
audio file. Then call

`python ucs_add_metadata.py --ucs_dir=dataset/raw/ucs_packname`

which will walk you through the process. For ucs files originally from the UCS
sharing site, with the same code name (e.g. 'CS280') some data can be automatically
found. Either way, some additional information will be needed like chart type,
level, etc. For the files downloaded from the ucs site, this metadata is gathered
automatically during the scraping process.

Note that ucs files with metadata cannot be guaranteed to work properly with other
programs like stepedit lite or various UCS viewing programs. If you wish to download the original
UCS files themselves only, you can add `--without_metadata` when running `ucs_scrape.py`.

## Processing the data

To convert 'ssc/ucs' files to usable format, there are several scripts in `processing/`.
First to extract data from the ssc/ucs files and convert to json format, you can call

`python processing/generate_json.py`

This will then ask you to choose a set of packs to convert from theose currently under 
`dataset/raw/`. The resulting json files will be located under `dataset/json/` in
directories according to their respective packs. These files will contain the data that can
be directly used during model training. If you modify certain pack folders under `dataset/raw`,
you may rerun the script, which will overwrite old json files with new ones.

## Creating a dataset

To specify a custom subset of charts you can create a dataset, comprised of potentially multiple packs.
This is done by calling `processing/build_dataset.py --dataset_name=name` which takes various filtering 
parameters, which include difficulty levels, chart type (single/double), song type (arcade/remix/...) 
and more. You may customize these parameters to build your own datasets (call `-h` for more details). This 
script will ask you which packs to choose songs and charts from as the initial layer of filtering.

The resulting file will be stored in `dataset/subsets/name.json` and will act as metadata
to be used when training is initiated for a particular model (see `train/`).
