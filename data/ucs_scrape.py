# automatically download UCS charts + metadata

import os
import argparse
import datetime

from spiders import ucs_spider, ucs_meta_spider
from ucs_add_metadata import META_JSON_PATH, crawl_base_download, crawl_meta_download
from scrapy.crawler import CrawlerProcess

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_artists', nargs='+', type=str, default=None,
        help='Restrict to ucs by certain step artists')
    parser.add_argument('--min_level', type=int, help='minimum difficulty level', default=1)
    parser.add_argument('--max_level', type=int, help='minimum difficulty level', default=28)
    parser.add_argument('--chart_type', type=str, choices=['single', 'double', 'coop',
        'douper', 'sinper'], help='type of chart')
    parser.add_argument('--songs', type=str, nargs='+', help='restrict to these songs', default=None)
    parser.add_argument('--start_date', type=str, help='restrict to ucs made since (YYYY-MM-DD)',
        default=datetime.date(year=2015, month=1, day=1))
    parser.add_argument('--end_date', type=str, help='restrict to ucs made up to (YYYY-MM-DD)',
        default=datetime.date.today())

    return parser.parse_args()

def get_date(date_arg):
    if isinstance(date_arg, datetime.date):
        return date_arg
    else:
        return datetime.date.fromisoformat(date_arg)

def main():
    args = parse_args()

    custom_pack_name = input('Please name this ucs pack: ')
    process = CrawlerProcess()

    # download base ucs/metadata if not yet already
    if not os.path.isdir(ucs_meta_spider.UCS_BASE_DATA_PATH):
        crawl_base_download()
    
    if not os.path.isfile(META_JSON_PATH):
        crawl_meta_download()

    start_date_obj = get_date(args.start_date)
    end_date_obj = get_date(args.end_date)

    process.crawl(ucs_spider.UCS_Spider, pack_name=custom_pack_name,
        step_artists=args.step_artists, min_level=args.min_level,
        max_level=args.max_level, chart_type=args.chart_type, songs=args.songs,
        start_date=start_date_obj, end_date=end_date_obj, meta_json_path=META_JSON_PATH)        
    process.start()

if __name__ == '__main__':
    main()