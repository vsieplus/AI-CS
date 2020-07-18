# This script converts a set of Pump it Up charts to basic json format
# based off of https://github.com/chrisdonahue/ddc/blob/master/dataset/extract_json.py

import json
import argparse
import os

from pathlib import Path

CHART_TYPES = ['ucs', 'ssc']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('packs_dir', type=str, help='directory of packs (organized like stepf2/p1 songs folder)')
    parser.add_argument('out_dir', type=str, help='output directory (will have same hierarchy as ^, but to store json files')

    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    packs_path = Path(args.packs_dir)
    pack_names = [pack for pack in packs_path.iterdir() if pack.is_dir()]


if __name__ == 'main':
    main()