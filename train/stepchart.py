# utility classes for loading step charts

import json
import os

from torch.utils.data import Dataset
from extract_audio_feats import load_audio, extract_audio_feats

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# potential caching https://discuss.pytorch.org/t/cache-datasets-pre-processing/1062/8

class StepchartDataset(Dataset):
    """Dataset of step charts"""

    # dataset_json: path to json file with dataset metadata
    def __init__(self, dataset_json):
        assert(os.path.isfile(dataset_json))
        with open(dataset_json, 'r') as f:
            self.metadata = json.loads(f.read())

        self.name = metadata['dataset_name']
        self.songtypes = metadata['songtypes']
        self.min_bpm = metadata['min_song_bpm']
        self.max_bpm = metadata['max_song_bpm']

        self.chart_type = metadata['chart_type']
        self.chart_authors = metadata['chart_authors']
        self.chart_difficulties = metadata['chart_difficulties']
        self.min_level = metadata['min_chart_difficulty']
        self.max_level = metadata['max_chart_difficulty']

        self.permutations = metadata['permutations']
        
        self.splits = metadata['splits']

        # the actual training samples
        self.load_charts(metadata['json_fps'])

    def __len__(self):
        return len(self.charts)

    def __getitem__(self, idx):
        return self.charts[idx]

    # filter/load charts
    def load_charts(self, json_fps):
        self.songs = []
        self.charts = []
        for fp in json_fps:
            chart = Chart(chart_attrs)
            

# returns splits of the given (torch) dataset; assumes 3 way split
def get_splits(dataset):
    split_sizes = []
    for i, split in enumerate(dataset.splits):
        if i == len(dataset.splits) - 1:
            split_sizes.append(len(full_dataset) - sum(split_sizes))
        else:
            split_sizes.append(split * len(full_dataset))
    train, valid, test = torch.utils.data.random_split(dataset, split_sizes)
    return train, valid, test

class Song(object):
    """A song object, corresponding to some audio file. May be relied on by
    multiple charts."""

    def __init__(self, audio_fp, title, artist):
        self.audio_fp = audio_fp
        self.title = title
        self.artist = artist

        waveform, sample_rate = load_audio(audio_fp)
        self.sample_rate = sample_rate

        # shape [3, ?, 80]
        self.audio_feats = extract_audio_feats(waveform, sample_rate)


class Chart(object):
    """A chart object, with associated data. Represents a single example"""
    CHART_PERMUTATIONS = {
        'pump-single': {
        #normal:         '01234'
            'flip':         '43210',
            'mirror':       '34201',
            'flip_mirror':  '10243'
        },

        'pump-double': {
            #normal:        '0123456789'
            'flip':         '9876543210',
            'mirror':       '9875643201',
            'flip_mirror':  '1023465789'
        }
    }

    def __init__(self):
        pass


    # https://github.com/chrisdonahue/ddc/blob/master/dataset/filter_json.py
    def add_permutations(chart_attrs):
        for chart in chart_attrs.get('charts'):
            chart['permutation'] = 'normal'

            chart_type = chart['stepstype']
            if chart_type == 'pump-routine':
                continue

            for permutation_name, permutation in CHART_PERMUTATIONS[chart_type].items():
                chart_copy = copy.deepcopy(chart)
                notes_cleaned = []
                for meas, beat, time, note in chart_copy['notes']:

                    # permutation numbers signify moved location
                    #   ex) note = '10010'
                    #       perm = '43210' -> (flip horizontally)
                    #       note_new = '01001'

                    note_new = ''.join([note[int(permutation[i])] for i in range(len(permutation))])

                    notes_cleaned.append((meas, beat, time, note_new))
                    chart_copy['notes'] = notes_cleaned
                    chart_copy['permutation'] = permutation_name

                chart_attrs['charts'].append(chart_copy)
        return chart_attrs