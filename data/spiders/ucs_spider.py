# Download specific ucs files along with metadata

import scrapy
from scrapy.pipelines.files import FilesPipeline

import re
import os
import datetime
from pathlib import Path

UCS_STEPARTIST_URL = 'http://www.piugame.com/bbs/board.php?ucs_event_no=&bo_table=ucs&sca=&sop=and&sfl=wr_name%2C1&stx='
UCS_SONG_URL = 'http://www.piugame.com/bbs/board.php?ucs_event_no=&bo_table=ucs&sca=&sop=and&sfl=ucs_song_no%2C1&stx='

ABS_PATH = pathlib.Path(__file__).parent.absolute()
OUT_DIR = os.path.join(str(ABS_PATH), 'dataset/raw/')

UCS_LVL_PATTERN = 's_lv([0-9][0-9])'

class UCS_Spider(scrapy.Spider):
    name = 'ucs_spider'

    custom_settings = {
        'ITEM_PIPELINES': {'spiders.ucs_spider.UCSPipeline': 1},
        'FILES_STORE': OUT_DIR,
        'MEDIA_ALLOW_REDIRECTS': True
    }

    # specific > generic (step_artist > song > level > mix > ...)
    def __init__(self, pack_name, step_artists, min_level, max_level,
        chart_type, songs, start_date, end_date):

        self.pack_name = pack_name
        self.min_level = min_level
        self.max_level = max_level
        self.chart_type = chart_type
        self.songs = songs

        self.start_date = start_date
        self.end_date = end_date

        assert(start_date < end_date)

        # if step artists/songs provided only start from their pages
        if step_artists:
            self.start_urls = [UCS_STEPARTIST_URL + step_artist for artist in step_artists]
            # may still have additional song restrictions
        elif songs:
            self.start_urls = [UCS_SONG_URL + song_title for song_title in songs]
        else:
            start_urls = ['http://www.piugame.com/bbs/board.php?bo_table=ucs']


    def parse(self, response):
        ucs_rows = response.xpath('//div[@class="share_list"]/table/tbody/tr')

        for ucs_row in ucs_rows:
            yield scrapy.Request(ucs_row, callback=self.parse_ucs_row)

        page_area = response.xpath('//span[@class="pg"]')

        curr_page = int(page_area.xpath('.//strong[@class="pg_current"]/text()').get())
        next_page = curr_page + 1
        next_page_link = page_area.xpath('.//a[contains(@href, "page=' + str(next_page) + '")]/@href').get()

        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse)

    def parse_ucs_row(self, response):
        # date in (YY-MM-DD) format
        upload_date = response.xpath('.//td[@class="share_upload_date"]/text()').get().split('-')
        upload_date_obj = date.datetime(year='20' + upload_date[0], 
            month=upload_date[1], day=upload_date[2])
        correct_date = upload_date_obj < self.start_date or self.end_date < upload_date_obj

        if not correct_date:
            return

        song_title = response.xpath('.//span[@class="share_song_title"]/a/text()').get()

        if not self.songs or self.songs and song_title in self.songs:
            chart_leveltype = response.xpath('.//td[@class="share_level"]/span/@class').get()
            chart_level = int(re.search(UCS_LVL_PATTERN, chart_leveltype).group(1))

            correct_chart_type = self.chart_type in chart_leveltype 
            valid_level = self.min_level <= chart_level and chart_level <= self.max_level

            if correct_chart_type and valid_level:
                ucs_page_url = response.xpath('.//span[@class="share_song_title"]/a/@href').get()
                yield response.follow(ucs_page_url, callback=self.parse_ucs_page,
                    meta={'chart_type': self.chart_type, 'level': chart_level})

    def parse_ucs_page(self, response):
        a = 1

# custom pipeline for ucs files
class UCSZipPipeline(FilesPipeline):
    def file_path(self, request, response=None, info=None):
        return request.meta.get('filename','')

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
            
            os.system('unzip -o -d ' + out_path + ' ' + fp_abs)
            os.system('rm -f ' + fp_abs)

        return item