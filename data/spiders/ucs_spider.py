# Download specific ucs files along with metadata

import scrapy
from scrapy.pipelines.files import FilesPipeline

import re
import os
import json
import datetime
from pathlib import Path
from getpass import getpass

from spiders.ucs_meta_spider import UCS_BASE_DATA_PATH

UCS_STEPARTIST_URL = 'http://www.piugame.com/bbs/board.php?ucs_event_no=&bo_table=ucs&sca=&sop=and&sfl=wr_name%2C1&stx='
UCS_SONG_URL = 'http://www.piugame.com/bbs/board.php?ucs_event_no=&bo_table=ucs&sca=&sop=and&sfl=ucs_song_no%2C1&stx='
POST_LOGIN_URL = 'https://www.piugame.com/bbs/piu.login_check.php'

ABS_PATH = Path(__file__).parent.absolute()
OUT_DIR = os.path.join(str(ABS_PATH), '../dataset/raw/')

UCS_LVL_PATTERN = 's_lv([0-9][0-9])'
UCS_CODE_PATTERN = '_(CS[0-9][0-9][0-9])#scrollSpeed='

class UCS_Spider(scrapy.Spider):
    name = 'ucs_spider'

    custom_settings = {
        'ITEM_PIPELINES': {'spiders.ucs_spider.UCSPipeline': 1},
        'FILES_STORE': OUT_DIR,
        'MEDIA_ALLOW_REDIRECTS': True
    }

    # specific > generic (step_artist > song > level > mix > ...)
    def __init__(self, pack_name, step_artists, min_level, max_level,
        chart_type, songs, start_date, end_date, meta_json_path):

        self.pack_name = pack_name
        outpath = os.path.join(OUT_DIR, pack_name)
        if not os.path.isdir(outpath):
            os.makedirs(outpath)

        self.min_level = min_level
        self.max_level = max_level
        self.chart_type = chart_type
        self.songs = songs

        self.start_date = start_date
        self.end_date = end_date

        with open(meta_json_path, 'r') as f:
            self.UCS_METADATA = json.loads(f.read())[0]

        assert(start_date < end_date)

        # if step artists/songs provided only start from their pages
        if step_artists:
            self.start_urls = [UCS_STEPARTIST_URL + artist for artist in step_artists]
            # may still have additional song restrictions
        elif songs:
            self.start_urls = [UCS_SONG_URL + song_title for song_title in songs]
        else:
            start_urls = ['http://www.piugame.com/bbs/board.php?bo_table=ucs']
        
        print('Please enter login credentials: ')
        self.user_email = input('User Email: ')
        self.user_pw = getpass('Password: ')
    
    def verify_login(self, response):
        return
        #if 'wrong Password' in response.xpath('//').get():
        #    self.logger.error('Login failed')

    def parse(self, response):
        # check if need to login or already logged in
        # http://scrapingauthority.com/2016/11/22/scrapy-login/
        login_area = response.xpath('//form[@name="foutlogin"]')

        if login_area:
            yield scrapy.FormRequest.from_response(response,
                formdata={'text': self.user_email, 'password': self.user_pw},
                callback=self.verify_login)

        ucs_rows = response.xpath('//div[@class="share_list"]/table/tbody/tr')

        for ucs_row in ucs_rows:
            for download in self.parse_ucs_row(ucs_row):
                yield download

        page_area = response.xpath('//span[@class="pg"]')

        curr_page = int(page_area.xpath('.//strong[@class="pg_current"]/text()').get())
        next_page = curr_page + 1
        next_page_link = page_area.xpath('.//a[contains(@href, "page=' + str(next_page) + '")]/@href').get()

        if next_page_link:
            yield response.follow(next_page_link, callback=self.parse)

    def parse_ucs_row(self, response):
        # date in (YY-MM-DD) format
        upload_date = response.xpath('.//td[@class="share_upload_date"]/text()').get().split('-')
        upload_date_obj = datetime.date(year=int('20' + upload_date[0]),
            month=int(upload_date[1]), day=int(upload_date[2]))
        correct_date = self.start_date < upload_date_obj and upload_date_obj < self.end_date

        if not correct_date:
            return

        song_title = response.xpath('.//span[@class="share_song_title"]/a/text()').get().strip().lower()

        if not self.songs or self.songs and song_title in self.songs:
            chart_leveltype = response.xpath('.//td[@class="share_level"]/span/@class').get()
            chart_level = int(re.search(UCS_LVL_PATTERN, chart_leveltype).group(1))

            correct_chart_type = self.chart_type in chart_leveltype 
            valid_level = self.min_level <= chart_level and chart_level <= self.max_level

            if correct_chart_type and valid_level:
                stepmaker = response.xpath('.//td[@class="share_stepmaker"]/text()').get().strip()
                ucs_code = re.search(UCS_CODE_PATTERN, response.get()).group(1)
                
                dl_link = response.xpath('.//a[@class="share_download2"]/@href').get()

                this_ucs_meta = self.UCS_METADATA[ucs_code]

                yield {**this_ucs_meta, **{'file_urls': [response.urljoin(dl_link)], 
                    'name': ucs_code, 'step_artist': stepmaker, 
                    'chart_type': self.chart_type, 'meter': self.chart_level,
                    'songtype': 'arcade', 'pack_name': self.pack_name}}


# custom pipeline for ucs files
class UCSPipeline(FilesPipeline):
    def file_path(self, request, response=None, info=None):
        return request.meta.get('filename','') + '.ucs'

    def get_media_requests(self, item, info):
        file_url = item['file_urls'][0]
        meta = {'filename': item['name']}
        yield scrapy.Request(url=file_url, meta=meta)
    
    # unzip upon download
    def item_completed(self, results, item, info):
        file_paths = [file['path'] for ok, file in results if ok]

        for fp in file_paths:
            # add metadata to file
            fp_abs = os.path.join(OUT_DIR, fp)
            add_metadata(item, fp_abs)

            # create + move to subfolder
            subfolder = os.path.join(OUT_DIR, item['pack_name'], item['title'])
            subfolder = os.path.join(subfolder, item['chart_type'] + item['meter']
                + '_' + item['step_artist'])
            os.makedirs(subfolder)

            os.system('mv {} {}'.format(fp_abs, subfolder))

        return item

    def add_metadata(self, item, fp):
        ucs_meta = ''
        for k,v in item.items():
            ucs_meta += ':{}={}'.format(k,v)

        with open(fp, 'r+') as f:
            old = f.read()
            f.seek(0,0)
            f.write(ucs_meta + old)