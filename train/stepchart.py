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
                    attrs['artist'], attrs['genre'], attrs['songtype'])

            for chart_idx in chart_indices:
                for permutation in self.permutations:
                    self.charts.append(Chart(attrs['charts'][chart_idx],
                        self.songs[song_name], orig_filetype, self.permutation))
    
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
                    valid_level = chart_level in self.chart_difficulties
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

    def __init__(self, audio_fp, title, artist, genre, songtype):
        self.audio_fp = audio_fp
        self.title = title
        self.artist = artist
        self.genre = genre
        self.songtype = songtype

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
AUDIO_FRAME_RATE = 100 # 10 ms per (audio) frame

UCS_SSC_DICT = {
    '.': '0',   # no step
    'X': '1',   # normal step
    'M': '2',   # start hold
    'H': '0',   # hold (0 between '2' ... '3' in ssc)
    'W': '3'    # release hold
}

SSC_OFF_SYMBOL = 0
SSC_STEP_SYMBOL = 1
SSC_HOLD_SYMBOL = 2
SSC_RELEASE_SYMBOL = 3
SSC_NUM_SYMBOLS = 4

class Chart(object):
    """A chart object, with associated data. Represents a single example"""
    
    # convert ucs steps to ssc (text)
    @classmethod
    def ucs_to_ssc(cls, steps):
        ssc_steps = ''
        for note in steps:
            ssc_steps += UCS_SSC_DICT[note]
        return ssc_steps

    # convert a sequence of steps ['00100', '10120', ...] -> input tensor
    @classmethod
    def sequence_to_tensor(cls, sequence):
        # shape [abs # of frames, 4 x # arrows (20 for single, 40 for double)]
        #   (for each arrow, mark 1 of 4 possible states - off, step, hold, release)
        # eg. ['10002', '01003'] -> [[0, 1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 1, 0] 
        #                             -downleft-   -upleft- ....   -downright-
        #                            [0, 0, 0, 0, 0, 1, 0, 0, ..., 0, 0, 0, 1]]
        step_tensors = []
        hold_indices = set()   # track active holds; for ssc, a '0' between '2' ... '3' is a hold
        for step in sequence:
            step_list = [int(symbol) for symbol in step]

            symbol_tensors = []
            for i, symbol in enumerate(step_list):            
                # treat hold starts ('2') as steps ('1'),
                if symbol == SSC_HOLD_SYMBOL:
                    hold_indices.add(symbol)
                    step_list[i] = SSC_STEP_SYMBOL
                elif i in hold_indices:
                    if symbol == SSC_RELEASE_SYMBOL:
                        hold_indices.remove(symbol)
                    # treat hold states ('0' between '2'..'3') as holds ('2')
                    elif symbol == SSC_OFF_SYMBOL:
                        step_list[i] = SSC_HOLD_SYMBOL

                symbol_tensors.append(torch.zeros(SSC_NUM_SYMBOLS).scatter_(0, symbol, 1))

            # convert symbols -> concatenated one hot encodings
            step_tensors.append(torch.cat(symbol_tensors))

        return torch.cat(step_tensors).view(-1, SSC_NUM_SYMBOLS)

    def __init__(self, chart_attrs, song, filetype, permuatation_type=None):
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
        self.permuatation_type = permutation_type

        self.parse_notes()

    def parse_notes(self):
        # [# frames] - numbers of (10ms) frames for each step
        self.step_frames = []

        # [# frames] for each frame in ^, whether or not a step was placed or not
        self.step_placements = []

        # [# frames, # step features] - sequence of non-empty steps with associated frame numbers
        self.step_sequence = []
        
        for _, _, time, steps in self.notes
            # list containing absolute frame numbers corresponding to each split
            int(round(time * AUDIO_FRAME_RATE))
            self.step_frames.append(curr_frame_number)
        
            # for each frame, 0 = no step, 1 = some step
            step_this_frame = STEP_PATTERNS[self.filetype].search(steps)
            self.step_placements.append(step_this_frame)

            self.step_sequence.append(self.permute_steps(steps))
        
        self.step_sequence = Chart.sequence_to_tensor(self.step_sequence)

    # return tensor of audio feats for this chart
    def get_audio_feats(self):
        return self.song.audio_feats

    def permute_steps(self, steps, permutation_type):  
        # permutation numbers signify moved location of original step
        #   ex) steps = '10010'
        #       permt = '43210' -> (flip horizontally)
        #    newsteps = '01001'            
        if self.filetype == 'ucs':
            steps = Chart.ucs_to_ssc(steps)

        if not self.permutation_type:
            return steps

        permutation = CHART_PERMUTATIONS[self.chart_type][self.permutation_type]

        return ''.join([steps[int(permutation[i])] for i in range(len(permutation))])
