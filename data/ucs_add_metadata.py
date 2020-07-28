# add metadata to ucs files, save as *.mucs

import argparse
import json
import pathlib
import glob
import os

from spiders import ucs_meta_spider
from scrapy.crawler import CrawlerProcess

ABS_PATH = pathlib.Path(__file__).parent.absolute()
CRAWLER_PATH = os.path.join(str(ABS_PATH), 'spiders/ucs_meta_spider.py')
META_JSON_PATH = os.path.join(str(ABS_PATH), 'dataset/00_ucs_metadata.json')

def add_metadata(ucs_dir, ucs_metadata):
    ucs_fps = glob.glob(os.path.join(ucs_dir, '*.ucs'))

    for ucs_fp in ucs_fps:
        ucs_id = os.path.split(ucs_fp)[-1].split('.')[0]

        try:
            ucs_meta = ucs_metadata[ucs_id]
        except KeyError:
            print('UCS code {} not found. Please enter additional info. '.format(ucs_id))
            ucs_meta = {}
            
            ucs_meta['title'] = input('Song name: ')
            ucs_meta['artist'] = input('Artist: ')
            ucs_meta['bpms'] = input('BPM: ')
            ucs_meta['version'] = input('Version debuted: ')
            
        # prompt user for other + chart-specific info
        print('Please enter the following chart information')
        ucs_meta['songtype'] = input('Song Type [arcade, remix, fullsong, shortcut]: ')
        ucs_meta['genre'] = input('Genre [k-pop, original, world_music, j-music, xross]: ')
        ucs_meta['author'] = input('Chart author: ')
        ucs_meta['chart_type'] = input('Chart type ["pump-single"/"pump-double"]:')
        ucs_meta['meter'] = input('Chart level: ')

        ucs_meta_str = ''
        for k,v in ucs_meta.items():
            ucs_meta_str += ':{}={}\n'.format(k, v)

        with open(ucs_fp, 'r+') as f:
            old = f.read()
            f.seek(0, 0)
            f.write(ucs_meta_str + old)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucs_dir', type=str, help='directory with base .ucs files', required=True)
    parser.add_argument('--scrape_meta', action='store_true', default=False, 
        help='scrape generic ucs metadata')
    parser.add_argument('--download', action='store_true', default=False,
        help='download UCS files from the site')

    args = parser.parse_args()

    # check if ucs metadata file exists, or if want to (re)scrape
    metajson_exists = os.path.isfile(META_JSON_PATH)
    if args.scrape_meta or not metajson_exists:
        if metajson_exists:
            os.system('rm ' + META_JSON_PATH)

        process = CrawlerProcess(settings={
            "FEEDS":{META_JSON_PATH: {"format": "json"}}
        })

        process.crawl(ucs_meta_spider.UCS_MetaSpider)
        process.start()

    if args.download or not os.path.isdir(ucs_meta_spider.UCS_BASE_DATA_PATH):
        process = CrawlerProcess()
        process.crawl(ucs_meta_spider.UCS_DownloadSpider)
        process.start()

    with open(META_JSON_PATH, 'r') as f:
        UCS_METADATA = json.loads(f.read())

    add_metadata(args.ucs_dir, UCS_METADATA)

if __name__ == '__main__':
    main()