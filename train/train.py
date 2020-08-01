# train a pair of models on the same dataset to perform step placement/selection

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, 
        help='Name of dataset under data/datasets/subsets/')
    parser.add_argument('--dataset_dir', type=str, default=None, 
        help='Alternatively, provide direct path to dataset')
    parser.add_argument('--output_dir', type=str, default=None, help="""Specify
        custom output directory to save models to. If blank, will save in
        ./models/dataset_name""")

    args = parser.parse_args()

    if not args.dataset_dir and args.dataset_name:
        args.dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name)
    
    if not args.output_dir:
        args.output_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_dir)[-1])

    return args

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

def main():
    args = parse_args()

    # Retrieve data

    # Train placement model

    # Train selection model


if __name__ == '__main__':
    main()