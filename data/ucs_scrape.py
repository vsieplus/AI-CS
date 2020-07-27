# scrape UCS charts

import os
import pathlib
import argparse

from spiders import ucs_spider
from scrapy.crawler import CrawlerProcess

UCS_URL = 'http://www.piugame.com/bbs/board.php?bo_table=ucs'

ABS_PATH = pathlib.Path(__file__).parent.absolute()
CRAWLER_PATH = os.path.join(str(ABS_PATH), 'spiders/ucs_spider.py')
OUT_DIR = os.path.join(str(ABS_PATH), 'dataset/raw/')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def main():
    args = parse_args()

    custom_packname = input('Please name this ucs pack:')
    out_path = os.path.join(OUT_DIR, custom_packname + '.json')

    process = CrawlerProcess(settings={
        "FEEDS":{out_path: {"format": "json"}}
    })

    process.crawl(ucs_meta_spider.UCS_METASPIDER, authors = args.authors,
        min_level = args.min_level) #...
    process.start()



if __name__ == '__main__':
    main()