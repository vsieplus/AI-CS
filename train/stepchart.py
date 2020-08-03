# utility classes for loading step charts

import json
import os

import torch
from torch.utils.data import Dataset, random_split

from extract_audio_feats import load_audio, extract_audio_feats

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# potential caching https://discuss.pytorch.org/t/cache-datasets-pre-processing/1062/8

# returns splits of the given (torch) dataset; assumes 3 way split
def get_splits(dataset):
    split_sizes = []
    for i, split in enumerate(dataset.splits):
        if i == len(dataset.splits) - 1:
            split_sizes.append(len(full_dataset) - sum(split_sizes))
        else:
            split_sizes.append(split * len(full_dataset))
    train, valid, test = random_split(dataset, split_sizes)
    return train, valid, test

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
        self.step_artists = metadata['step_artists']
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
        self.songs = {}
        self.charts = []
        for fp in json_fps:
            with open(fp, 'r') as f:
                attrs = json.loads(f.read())
            
            # check attrs for each chart to see if we should add it to the dataset
            chart_indices = self.charts_to_include(attrs)

            if not chart_indices:
                continue

            # .ssc (may contain multiple charts/same song) or .ucs (always 1 chart)    
            orig_filetype = attrs['chart_fp'].split('.')[-1]

            # create new song if needed
            song_name = attrs['title']
            if song_name not in self.songs:
                self.songs[song_name] = Song(attrs['music_fp'], song_name,
                    attrs['artist'], attrs['genre'], attrs['songtype'], attrs['bpms']) 
                    #TODO add diplay bpms to chart json outputs

            for chart_idx in chart_indices:
                self.charts.append(Chart(attrs['charts'][chart_idx],
                    self.songs[song_name], orig_filetype, self.permutations))
    
    # determine which charts in the given attrs belongs in this dataset
    # return list of indices, w/length between 0 <= ... <= len(attrs['charts'])
    def charts_to_include(self, attrs):
        chart_indices = []

        valid_songtype = self.songtypes and attrs['songtype'] not in self.songtypes
        valid_bpm = self.min_song_bpm <= min(attrs['display_bpm']) and
            max(attrs['display_bpm']) <= self.max_song_bpm

        if valid_songtype and valid_bpm:
            for i, chart_attrs in enumerate(attrs['charts']):
                valid_type = chart_attrs['stepstype'] == self.chart_type

                if not valid_type:
                    continue

                chart_level = chart_attrs['meter']
                if self.chart_difficulties:
                    valid_level = chart_level  in self.chart_difficulties
                else:
                    valid_level = self.min_chart_difficulty <= chart_level
                        and chart_level <= self.max_chart_difficulty

                if not valid_level:
                    continue

                valid_author = not self.step_artists or self.step_artists and
                    chart_attrs['credit'] in self.step_artists

                if not valid_author:
                    continue
                
                chart_indices.append(i)

        return chart_indices

class Song(object):
    """A song object, corresponding to some audio file. May be relied on by
    multiple charts."""

    def __init__(self, audio_fp, title, artist, genre, songtype, bpms):
        self.audio_fp = audio_fp
        self.title = title
        self.artist = artist
        self.genre = genre
        self.songtype = songtype
        self.bpms = bpms

        self.load_audio()

    def load_audio(self):
        waveform, sample_rate = load_audio(self.audio_fp)
        self.sample_rate = sample_rate

        # shape [3, ?, 80]
        self.audio_feats = extract_audio_feats(waveform, sample_rate)


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

STEP_PATTERNS = {
    'ucs': re.compile('[XMHW]'),
    'ssc': re.compile('[1-3]')
}

N_CHART_TYPES = 2
N_LEVELS = 28
MS_PER_FRAME = 100

UCS_SSC_DICT = {
    '.': '0',
    'X': '1',
    'M': '2',   # start hold
    'H': '0',   # hold
    'W': '3'    # release hold
}

class Chart(object):
    """A chart object, with associated data. Represents a single example"""
    
    # convert ucs steps to ssc
    def ucs_to_ssc(cls, steps):
        ssc_steps = ''
        for note in steps:
            ssc_steps += UCS_SSC_DICT[note]
        return ssc_steps

    # convert a sequence of steps ['00100', '10120', ...] -> input tensor
    def sequence_to_tensor(cls, sequence):
        pass

    def __init__(self, chart_attrs, song, filetype, permuatations):
        self.song = song
        self.filetype = filetype

        self.step_artist = chart_attrs['credit']
        self.level = chart_attrs['meter']
        self.chart_type = chart_attrs['stepstype']
        assert(self.chart_type in CHART_PERMUTATIONS)

        # concat one-hot encodings of chart_type/level
        chart_type_onehot = torch.zeros(N_CHART_TYPES).scatter_(0,
            1 if self.chart_type == "pump-double" else 0, 1)
        level_oneshot = torch.zeros(N_LEVELS).scatter_(0, self.level - 1, 1)
        self.chart_feats = torch.cat((chart_type_onehot, level_oneshot), dim=-1)

        self.notes = chart_attrs['notes']
        self.permutations = permutations

        self.parse_notes()

    def parse_notes(self):
        self.step_frames = []
        self.step_placements = []

        # track non-empty steps + relevant features
        self.step_sequences = [[] for _ in range(len(self.permutations))]
        
        for _, _, time, steps in self.notes
            # list containing absolute frame numbers corresponding to each split
            self.step_frames.append(int(round(time * MS_PER_FRAME)))
        
            # for each frame, 0 = no step, 1 = some step
            step_this_frame = STEP_PATTERNS[self.filetype].search(steps)
            self.step_placements.append(step_this_frame)

            for i, permutation_type in enumerate(self.permutations):
                self.step_sequences[i].append(self.permute_steps(steps, permutation_type))
        
        self.step_frames = torch.tensor(self.step_frames)
        self.step_placements = torch.tensor(self.step_placements, dtype=torch.uint8)
        self.step_sequences = [Chart.sequence_to_tensor(seq) for seq in self.step_sequences]

    # return tensor of audio feats for this chart
    def get_audio_feats(self):
        return self.song.audio_feats

    # https://github.com/chrisdonahue/ddc/blob/master/dataset/filter_json.py
    def permute_steps(self, steps, permutation_type):  
        # permutation numbers signify moved location
        #   ex) steps = '10010'
        #       perm = '43210' -> (flip horizontally)
        #       steps_new = '01001'
        permutation = CHART_PERMUTATIONS[self.chart_type][permutation_type]

        if self.filetype == 'ucs':
            steps = Chart.ucs_to_ssc(steps)

        return ''.join([steps[int(permutation[i])] for i in range(len(permutation))])