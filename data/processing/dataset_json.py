# 'Construct' custom subsets of data - represent as text files with json filepaths,
# in the same directory as a separate json with dataset metadata
# adapted from https://github.com/chrisdonahue/ddc/blob/master/dataset/dataset_json.py

import argparse
import os
import json
import random
from util import get_subdirs, ez_name

from pathlib import Path

ABS_PATH = Path(__file__).parent.absolute()
DEFAULT_JSON_PATH = os.path.join(str(ABS_PATH), '../dataset/json')
DEFAULT_DATASETS_PATH = os.path.join(str(ABS_PATH), '../dataset/subsets')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, help='Input JSON dir')
    parser.add_argument('--datasets_dir', type=str, help='Dir to output file containing dataset info')
    parser.add_argument('--splits', type=str, help='CSV list of split values for datasets (e.g. 0.8,0.1,0.1)')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='If set, shuffle dataset before split')
    parser.add_argument('--shuffle_seed', type=int, help='If set, use this seed for shuffling')
    parser.add_argument('--choose', dest='choose', action='store_true', help='If set, choose from list of packs')

    # song-filtering
    parser.add_argument('--song_types', type=str, help='song types to include; if empty no filter',
        choices = ['arcade', 'fullsong', 'remix', 'shortcut'], nargs='+')

    # chart-filtering
    parser.add_argument('--chart_type', type=str, help='pick chart type for this dataset',
        choices = ['pump-single', 'pump-double'], required=True)
    parser.add_argument('--step_artists', type=str, help='step chart authors to include if known, no filter if empty',
        nargs='+')
    parser.add_argument('--chart_difficulties', type=str, help='Whitelist of chart difficulties; if empty, no filter')
    parser.add_argument('--min_difficulty', type=int, help='Min chart difficulty')
    parser.add_argument('--max_difficulty', type=int, help='Max chart difficulty')
    parser.add_argument('--permutations', type=str, help='List of permutation types to include in output',
        choices = ['flip', 'mirror', 'flip_mirror'], nargs='+') # for data augmentation

    parser.set_defaults(
        json_dir=DEFAULT_JSON_PATH,
        datasets_dir=DEFAULT_DATASETS_PATH,
        step_artists=[],
        chart_difficulties=[],
        choose=True)
    
    return parser.parse_args()

def display_args(args):
    print('\nIncluding song types:', args.song_types)
    print('Using chart type:', args.chart_type)
    if(len(args.step_artists) > 0):
        print('Limiting to chart authors:', args.step_artists)
    if(len(args.chart_difficulties) > 0):
        print('Using chart difficulties:', args.chart_difficulties)
    print('Min. chart difficulty:', args.min_difficulty)
    print('Max. chart difficulty:', args.max_difficulty)
    print('Using chart permutations:', args.permutations)

def main():
    args = parse_args()
    dataset_name = input('Please enter a name for this dataset: ')
    dataset_name = ez_name(dataset_name)

    if not os.path.isdir(args.datasets_dir):
        os.makedirs(args.datasets_dir)

    splits = [float(x) for x in args.splits.split(',')]

    if len(splits) == 0:
        splits = [1.0]
    else:
        splits = [x / sum(splits) for x in splits]

    pack_names = get_subdirs(args.json_dir, args.choose)

    display_args(args)

    json_fps = []
    for pack_name in pack_names:
        pack_dir = os.path.join(args.json_dir, pack_name)
        sub_fps = os.listdir(pack_dir)
        sub_fps = [os.path.abspath(os.path.join(pack_dir, sub_fp)) for sub_fp in sub_fps]

        if args.shuffle:
            random.seed(args.shuffle_seed)
            random.shuffle(sub_fps)

        json_fps.extend(sub_fps)

    if not args.step_artists:
        known_step_artists = set()
        for fp in json_fps:
            with open(fp, 'r') as f:
                chart_data = json.loads(f.read())
            for chart in chart_data['charts']:
                if 'credit' in chart:
                    known_step_artists.add(chart['credit'])
        known_step_artists = list(known_step_artists)

    dataset_metadata = {
        'dataset_name': dataset_name,
        'song_types': args.song_types,
        'mixes': pack_names,
        'chart_type': args.chart_type,
        'step_artists': args.step_artists,
        'chart_difficulties': args.chart_difficulties,
        'min_chart_difficulty': args.min_difficulty,
        'max_chart_difficulty': args.max_difficulty,
        'permutations': args.permutations,
        'splits': splits,
        'json_fps': json_fps
    }

    dataset_json = os.path.join(args.datasets_dir, dataset_name + '.json')
    with open(dataset_json, 'w') as f:
        f.write(json.dumps(dataset_metadata, indent=2))
    
    print('Dataset "{}" saved to {}'.format(dataset_name, os.path.relpath(dataset_json)))
    print('Total number of chart files: ', len(json_fps))
    print('Included known step artists: ', known_step_artists)

if __name__ == '__main__':
    main()
    