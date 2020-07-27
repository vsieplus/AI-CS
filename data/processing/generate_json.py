# This script converts a set of Pump it Up charts (ssc/ucs) to basic json format
# based off of https://github.com/chrisdonahue/ddc/blob/master/dataset/extract_json.py

import json
import argparse
import os
import logging
import glob
import copy

from collections import OrderedDict
from pathlib import Path

import parse
from util import get_subdirs, ez_name

ABS_PATH = Path(__file__).parent.absolute()
DEFAULT_OUT_PATH = os.path.join(str(ABS_PATH), '../dataset/json')

CHART_TYPES = ['ucs', 'ssc']
UCS_PACKNAME = 'UCS'

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
    parser.add_argument('--data_dir', type=str, 
        help='directory of packs (organized like stepf2/p1 songs folder)',
        default=str(ABS_PATH) + '/../dataset/raw')
    parser.add_argument('--out_dir', type=str, help='output directory',
        default = DEFAULT_OUT_PATH)
    parser.add_argument('--choose', action='store_true', help='select specific packs', default=False)
    
    return parser.parse_args()

def search_for_music(chart_filename, root):
    """Try to find a music file for the corresponding chart file"""

    for filename in os.listdir(root):
        prefix, ext = os.path.splitext(filename)
        if ext.lower()[1:] in ['mp3', 'ogg']:
            music_names.append(filename)

    if len(music_names) == 0:
        raise ValueError('no audio file found for {}'.format(chart_filename))

    return music_names[0]

# https://github.com/chrisdonahue/ddc/blob/master/dataset/filter_json.py
def add_permutations(chart_attrs):
    for chart in chart_attrs.get('charts'):
        chart['permutation'] = 'normal'

        for permutation_name, permutation in CHART_PERMUTATIONS[chart['stepstype']].items():
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

def parse_pack_charts(pack_name_clean, pack_chart_files, chart_type, out_dir):
    """Parses a set of pack's chart files (ucs/ssc)"""

    assert(chart_type in CHART_TYPES)

    if len(pack_chart_files) > 0:
        pack_outdir = os.path.join(out_dir, pack_name_clean)
        if not os.path.isdir(pack_outdir):
            os.mkdir(pack_outdir)
        
    chart_files_clean = set()
    for pack_chart_file in pack_chart_files:
        chart_filename = os.path.split(os.path.split(pack_chart_file)[0])[1]
        chart_filename_clean = ez_name(chart_filename)
        if chart_filename_clean in chart_files_clean:
            raise ValueError('song name conflict: {}'.format(chart_filename_clean))
        chart_files_clean.add(chart_filename_clean)

        with open(pack_chart_file, 'r') as f:
            chart_txt = f.read()

        # parse chart metadata from the text for the given chart type
        try:
            chart_attrs = parse.parse_chart_txt(chart_txt, chart_type)
        except Exception as e:
            logging.error('{} in\n{}'.format(e, pack_chart_file))
            raise e

        # store music path
        root = os.path.abspath(os.path.join(pack_chart_file, '..'))
        music_fp = os.path.join(root, chart_attrs.get('music', ''))

        if 'music' not in chart_attrs or not os.path.exists(music_fp):
            try:
                music_fp = search_for_music(os.path.splitext(chart_filename)[0], root)
            except ValueError as e:
                continue

        chart_attrs = add_permutations(chart_attrs)

        # constrcut json object to save with important fields
        out_json_path = os.path.join(pack_outdir, '{}_{}.json'.format(pack_name_clean, chart_filename_clean))
        out_json = OrderedDict([
            ('chart_fp', os.path.abspath(pack_chart_file)),
            ('music_fp', os.path.abspath(music_fp)),
            ('pack', pack_name_clean),
            ('title', chart_attrs.get('title')),
            ('artist', chart_attrs.get('artist')),
            ('genre', chart_attrs.get('genre')),
            ('songtype', chart_attrs.get('songtype')),      # arcade/remix/shortcut/...
            ('charts', chart_attrs.get('charts'))
        ])

        with open(out_json_path, 'w') as out_f:
            try:
                out_f.write(json.dumps(out_json))
            except UnicodeDecodeError:
                logging.error('Unicode error in {}'.format(pack_chart_file))
                continue

            print('Parsed {} - {}: {} charts'.format(pack_name_clean, 
                chart_filename, len(out_json['charts'])))
    
def main():
    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # retrieve pack names, 
    packs_path = args.data_dir
    pack_names = get_subdirs(packs_path, args.choose)
    
    pack_names_clean = set()

    for pack_name in pack_names:
        pack_name_clean = ez_name(pack_name)
        if pack_name_clean in pack_names_clean:
            raise ValueError('pack name conflict: {}'.format(pack_name_clean))
        pack_names_clean.add(pack_name_clean)

        pack_dir = os.path.join(args.data_dir, pack_name)
        
        if pack_name == UCS_PACKNAME:
            chart_type = 'ucs'
        else:
            chart_type = 'ssc'

        pack_globs = os.path.join(pack_dir, '*', '*.' + chart_type)
        pack_chart_files = sorted(glob.glob(pack_globs))

        parse_pack_charts(pack_name_clean, pack_chart_files, chart_type, args.out_dir)
            

if __name__ == '__main__':
    main()