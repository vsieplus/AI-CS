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
    parser.add_argument('--datasets_dir', type=str, help='Dir to output text file containing dataset info')
    parser.add_argument('--splits', type=str, help='CSV list of split values for datasets (e.g. 0.8,0.1,0.1)')
    parser.add_argument('--splitnames', type=str, help='CSV list of split names for datasets (e.g. train,test,eval)')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help='If set, shuffle dataset before split')
    parser.add_argument('--shuffle_seed', type=int, help='If set, use this seed for shuffling')
    parser.add_argument('--choose', dest='choose', action='store_true', help='If set, choose from list of packs')

    # song-filtering
    parser.add_argument('--song_types', type=str, help='song types to incluce; if empty no filter',
        choices = ['arcade', 'fullsong', 'remix', 'shortcut'], nargs='+')

    # chart-filtering
    parser.add_argument('--chart_type', type=str, help='pick chart type for this dataset',
        choices = ['pump-single', 'pump-double'])
    parser.add_argument('--chart_authors', type=str, help='step chart authors to include if known, no filter if empty',
        nargs='+')
    parser.add_argument('--chart_difficulties', type=str, help='Whitelist of chart difficulties; if empty, no filter')
    parser.add_argument('--min_difficulty', type=int, help='Min chart difficulty')
    parser.add_argument('--max_difficulty', type=int, help='Max chart difficulty')
    parser.add_argument('--min_bpm', type=int, help='Minimum song bpm')
    parser.add_argument('--max_bpm', type=int, help='Maximum song bpm')
    parser.add_argument('--permutations', type=str, help='List of permutation types to include in output',
        choices = ['flip', 'mirror', 'flip_mirror'], nargs='+')

    parser.set_defaults(
        json_dir=DEFAULT_JSON_PATH,
        datasets_dir=DEFAULT_DATASETS_PATH,
        chart_authors=[],
        chart_difficulties=[],
        choose=True)
    
    return parser.parse_args()

def display_args(args):
    print('Including song types:', args.song_types)
    print('Using chart type:', args.chart_type)
    if(len(args.chart_authors) > 0):
        print('Limiting to chart authors:', args.chart_authors)
    if(len(args.chart_difficulties) > 0):
        print('Using chart difficulties:', args.chart_difficulties)
    print('Min. chart difficulty:', args.min_difficulty)
    print('Max. chart difficulty:', args.max_difficulty)
    print('Using chart permutations:', args.permutations)

def main():
    args = parse_args()
    dataset_name = input('Please enter a name for this dataset: ')
    dataset_name = ez_name(dataset_name)

    display_args(args)

    output_dir = os.path.join(args.datasets_dir, dataset_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    splits = [float(x) for x in args.splits.split(',')]
    split_names = [x.strip() for x in args.splitnames.split(',')]
    assert len(splits) == len(split_names)

    pack_names = get_subdirs(args.json_dir, args.choose)

    dataset_metadata = {
        'dataset_name': dataset_name,
        'song_types': args.song_types,
        'mixes': pack_names,
        'min_song_bpm': args.min_bpm,
        'max_song_bpm': args.max_bpm,
        'chart_type': args.chart_type,
        'chart_authors': args.chart_authors,
        'chart_difficulties': args.chart_difficulties,
        'min_chart_difficulty': args.min_difficulty,
        'max_chart_difficulty': args.max_difficulty,
        'permutations': args.permutations    
    }

    dataset_json = os.path.join(output_dir, dataset_name + '.json')
    with open(dataset_json, 'w') as f:
        f.write(json.dumps(dataset_metadata, indent=4))

    for pack_name in pack_names:
        pack_dir = os.path.join(args.json_dir, pack_name)
        sub_fps = sorted(os.listdir(pack_dir))
        sub_fps = [os.path.abspath(os.path.join(pack_dir, sub_fp)) for sub_fp in sub_fps]

        if args.shuffle:
            random.seed(args.shuffle_seed)
            random.shuffle(sub_fps)
        
        if len(splits) == 0:
            splits = [1.0]
        else:
            splits = [x / sum(splits) for x in splits]

        split_ints = [int(len(sub_fps) * split) for split in splits]
        split_ints[0] += len(sub_fps) - sum(split_ints)

        split_fps = []
        for split_int in split_ints:
            split_fps.append(sub_fps[:split_int])
            sub_fps = sub_fps[split_int:]

        for split, splitname in zip(split_fps, split_names):
            out_name = '{}{}.txt'.format(pack_name, '_' + splitname if splitname else '')
            out_fp = os.path.join(output_dir, out_name)
            with open(out_fp, 'w') as f:
                f.write('\n'.join(split))
    
    print('Dataset "{}" saved to {}'.format(dataset_name, os.path.relpath(output_dir)))

if __name__ == '__main__':
    main()
    