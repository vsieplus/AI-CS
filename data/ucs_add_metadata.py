# add metadata to ucs files, save as *.mucs

import argparse
import json
import pathlib
import glob

from spiders import ucs_meta_spider
from scrapy.crawler import CrawlerProcess

ABS_PATH = pathlib.Path(__file__).parent.absolute()
CRAWLER_PATH = os.path.join(str(ABS_PATH), 'spiders/ucs_meta_spider.py')
META_JSON_PATH = os.path.join(str(ABS_PATH), 'ucs_metadata.json')

def add_metadata(ucs_dir, ucs_metadata):
    ucs_fps = glob.glob(os.path.join(ucs_dir, '*.ucs'))

    for ucs_fp in ucs_fps:
        ucs_id = ucs_fp.split('.')[0]

        try:
            ucs_data = UCS_METADATA[ucs_id]
        except KeyError:
            print('UCS code {} not found. Please enter additional info:'.format(ucs_id))


def main():
    parser = argparser.ArgumentParser()
    parser.add_argument('--ucs_dir', type=str, help='directory with base .ucs files', required=True)
    parser.add_argument('--scrape_meta', action='store_true', default=False, 
        help='scrape generic ucs metadata')

    args = parser.parse_args()

    # check if ucs metadata file exists, or if want to (re)scrape
    if args.scrape_meta or not os.isfile(META_JSON_PATH):
        process = CrawlerProcess(settings={
            "FEEDS":{META_JSON_PATH: {"format": "json"}}
        })

        process.crawl(ucs_meta_spider.UCS_METASPIDER)
        process.start()

    with open(META_JSON_PATH, 'r') as f:
        UCS_METADATA = json.loads(f.read())

    add_metadata(args.ucs_dir, UCS_METADATA)

if __name__ == '__main__':
    main()