# Scrape ucs metadata and base files from the UCS site

import argparse
import os
import pathlib

from spiders import ucs_meta_spider

import scrapy
from scrapy.crawler import CrawlerProcess

ABS_PATH = pathlib.Path(__file__).parent.absolute()
META_JSON_PATH = os.path.join(str(ABS_PATH), 'dataset/00_ucs_metadata.json')

def crawl_base_download():
    process = CrawlerProcess()
    process.crawl(ucs_meta_spider.UCS_BaseDownloadSpider)
    process.start()

def crawl_meta_download(stop_after):
    process = CrawlerProcess(settings={
        'FEED_URI': META_JSON_PATH,
        'FEED_EXPORTERS': { 'jsonlines': 'scrapy.exporters.JsonItemExporter'},
        'FEED_FORMAT': 'json',
        'FEED_EXPORT_BEAUTIFY': True,
        'FEED_EXPORT_INDENT': 2
    })

    process.crawl(ucs_meta_spider.UCS_MetaSpider)
    process.start(stop_after_crawl=stop_after)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scrape_meta', action='store_true', default=False, help='scrape generic ucs metadata')
    parser.add_argument('--download', action='store_true', default=False, help='download UCS files from the site')

    args = parser.parse_args()

    if args.scrape_meta:
        # remove old scraped data
        if os.path.isfile(META_JSON_PATH):
            os.system('rm ' + META_JSON_PATH)

        print('Scraping ucs metadata')
        crawl_meta_download(stop_after=not args.download)

    if args.download:
        print('Downloading base ucs files')
        crawl_base_download()

if __name__ == '__main__':
    main()