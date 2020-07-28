# scrape ucs metadatas

import scrapy
from scrapy.pipelines.files import FilesPipeline

import re
import os
from pathlib import Path

ABS_PATH = Path(__file__).parent.absolute()
UCS_BASE_DATA_PATH = os.path.join(str(ABS_PATH), '../dataset/raw/00-UCS_BASE')

if not os.path.isdir(UCS_BASE_DATA_PATH):
    os.makedirs(UCS_BASE_DATA_PATH)

VERSION_PATTERN = re.compile('version_([0-9]+)')
VERSION_DICT = {
    1: '1st', 2: '2nd', 3: 'obg_3rd', 4: 'obg_season_evolution',
    5: 'perfect_collection', 6: 'extra', 7: 'prex', 8: 'premiere',
    9: 'rebirth', 10: 'exceed', 11: 'exceed2', 12: 'zero', 13: 'nx', 
    14: 'nx2', 15: 'nx_absolute', 16: 'fiesta', 17: 'fiesta_ex', 
    18: 'fiesta2', 19: 'prime', 20: 'prime2', 21: 'xx'}

# crawl ucs file downloads
class UCS_DownloadSpider(scrapy.Spider):
    name = 'ucs_download'
    start_urls = ['http://www.piugame.com/piu.ucs/ucs.sample/ucs.sample.alltunes.php']
       
    custom_settings = {
        'ITEM_PIPELINES': {'spiders.ucs_meta_spider.UCSZipPipeline': 1},
        'FILES_STORE': UCS_BASE_DATA_PATH,
        'MEDIA_ALLOW_REDIRECTS': True
    }

    def parse(self, response):
        ucs_rows = response.xpath('//tr')
        ucs_codes = ucs_rows.xpath('.//span[@class="download_cs_number"]/text()').getall()
        
        # download/unzip ucs zip files
        dl_links = ucs_rows.xpath('.//td[@class="download_source"]').xpath('.//a/@href').getall()
        assert(len(ucs_codes) == len(dl_links))

        for i, dl_link in enumerate(dl_links):
            yield {'file_urls': [response.urljoin(dl_link)], 'name': ucs_codes[i]}

        page_area = response.xpath('//span[@class="pg"]')

        curr_page = int(page_area.xpath('.//strong[@class="pg_current"]/text()').get())
        next_page = curr_page + 1
        next_page_link = page_area.xpath('.//a[contains(@href, "' + str(next_page) + '")]/@href').get()

        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse)

# crawl ucs metadata
class UCS_MetaSpider(scrapy.Spider):
    name = 'ucs_meta'
    start_urls = ['http://www.piugame.com/piu.ucs/ucs.sample/ucs.sample.alltunes.php']
    
    def parse(self, response):
        # //tr for each row corresponding to a ucs
        ucs_rows = response.xpath('//tr')

        ucs_codes = ucs_rows.xpath('.//span[@class="download_cs_number"]/text()').getall()
        song_names = ucs_rows.xpath('.//span[@class="list_song_title "]/text()').getall()
        song_artists = ucs_rows.xpath('.//span[@class="list_song_artist"]/text()').getall()
        song_artists = [re.sub('/ ', '', artist_str) for artist_str in song_artists]

        bpms = ucs_rows.xpath('.//td[@class="download_bpm"]/text()').getall()
        versions = [ver.extract() for ver in ucs_rows.css('.version_img').xpath('@class')]
        versions = [int(VERSION_PATTERN.search(ver).group(1)) for ver in versions]
        versions = [VERSION_DICT[ver] for ver in versions]

        assert(len(versions) == len(bpms) and len(bpms) == len(song_artists) and
            len(song_artists) == len(song_names) and len(song_names) == len(ucs_codes))

        # add ucs data to ucs_dict
        if 'ucs_dict' in response.meta:
            ucs_dict = response.meta['ucs_dict']
        else:
            ucs_dict = {}

        for i, code in enumerate(ucs_codes):
            ucs_meta = {}
            ucs_meta['title'] = song_names[i]
            ucs_meta['artist'] = song_artists[i]
            ucs_meta['bpms'] = bpms[i]
            ucs_meta['version'] = versions[i]

            ucs_dict[code] = ucs_meta

        # check for next page
        page_area = response.xpath('//span[@class="pg"]')

        curr_page = int(page_area.xpath('.//strong[@class="pg_current"]/text()').get())
        next_page = curr_page + 1
        next_page_link = page_area.xpath('.//a[contains(@href, "' + str(next_page) + '")]/@href').get()

        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse, 
                meta={**(response.meta), **{'ucs_dict': ucs_dict}})
        else:
            yield ucs_dict      

class UCSZipPipeline(FilesPipeline):    
    def file_path(self, request, response=None, info=None):
        return request.meta.get('filename','') + '.zip'

    def get_media_requests(self, item, info):
        file_url = item['file_urls'][0]
        meta = {'filename': item['name']}
        yield scrapy.Request(url=file_url, meta=meta)
    
    # unzip upon download
    def item_completed(self, results, item, info):
        file_paths = [file['path'] for ok, file in results if ok]

        for fp in file_paths:
            fp_abs = os.path.join(UCS_BASE_DATA_PATH, fp)

            out_path = re.sub('.zip', '', fp_abs)
            
            os.system('unzip -d ' + out_path + ' ' + fp_abs)
            os.system('rm -f ' + fp_abs)

        return item