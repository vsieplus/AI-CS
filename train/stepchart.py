# utility classes for loading step charts

import json
import os

from torch.utils.data import Dataset

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class StepchartDataset(Dataset):
    """Dataset of step charts"""

    # dataset_json: path to json file with dataset metadata
    def __init__(self, dataset_json):
        assert(os.path.isfile(dataset_json))
        with open(dataset_json, 'r') as f:
            self.metadata = json.loads(f.read()) 

        self.num_charts = 1

    def __len__(self):
        return self.num_charts

    def __getitem__(self, idx):
        pass

# below adapted from https://github.com/chrisdonahue/ddc/blob/master_v2/ddc/chart.py
class Song(object):
    """A song object, corresponding to some audio file"""

    def __init__(self, audio_fp):
        pass


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

class Chart(object):
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