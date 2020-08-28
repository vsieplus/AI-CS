# add metadata to ucs files

import argparse
import json
import pathlib
import glob
import os

from spiders import ucs_meta_spider
from scrapy.crawler import CrawlerProcess

ABS_PATH = pathlib.Path(__file__).parent.absolute()
META_JSON_PATH = os.path.join(str(ABS_PATH), 'dataset/00_ucs_metadata.json')

def add_metadata(ucs_dir, ucs_metadata):
    ucs_fps = glob.glob(os.path.join(ucs_dir, '*/*.ucs'))
    ucs_fps.extend(glob.glob(os.path.join(ucs_dir, '*/*/*.ucs')))

    for ucs_fp in ucs_fps:
        add_meta = input('Add metadata to {}? (y/n) '.format(ucs_fp))

        if add_meta != 'y':
            continue

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
        print('Please enter the following chart information for {}'.format(ucs_fp))
        ucs_meta['songtype'] = input('Song Type [arcade, remix, fullsong, shortcut]: ')
        ucs_meta['genre'] = input('Genre [k-pop, original, world-music, j-music, xross]: ')
        ucs_meta['step_artist'] = input('Step artist: ')
        ucs_meta['chart_type'] = input('Chart type ["pump-single"/"pump-double"]: ')
        ucs_meta['meter'] = input('Chart level: ')

        ucs_meta_str = ''
        for k,v in ucs_meta.items():
            ucs_meta_str += ':{}={}\n'.format(k, v)

        # prepend the ucs metadata
        with open(ucs_fp, 'r+') as f:
            old = f.read()
            f.seek(0, 0)
            f.write(ucs_meta_str + old)

        print("Added metadata to {}\n".format(ucs_fp))

def crawl_base_download():
    process = CrawlerProcess()
    process.crawl(ucs_meta_spider.UCS_BaseDownloadSpider)
    process.start()

def crawl_meta_download():
    process = CrawlerProcess(settings={"FEEDS":{META_JSON_PATH: {"format": "json"}}})

    process.crawl(ucs_meta_spider.UCS_MetaSpider)
    process.start(stop_after_crawl=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucs_dir', type=str, help='directory with base .ucs files', default=None)
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

        crawl_meta_download()

    if args.download or not os.path.isdir(ucs_meta_spider.UCS_BASE_DATA_PATH):
        crawl_base_download()

    with open(META_JSON_PATH, 'r') as f:
        UCS_METADATA = json.loads(f.read())[0]

    if args.ucs_dir:
        add_metadata(args.ucs_dir, UCS_METADATA)

if __name__ == '__main__':
    main()
