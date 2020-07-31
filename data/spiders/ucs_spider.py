# Download specific ucs files along with metadata

import scrapy
from lxml import html
from lxml.etree import tostring

import re
import os
import requests
import json
import datetime
from pathlib import Path
from getpass import getpass

from spiders.ucs_meta_spider import UCS_BASE_DATA_PATH

ABS_PATH = Path(__file__).parent.absolute()

import sys
sys.path.append(os.path.join(str(ABS_PATH), "../processing"))
from util import ez_name

UCS_STEPARTIST_URL = 'http://www.piugame.com/bbs/board.php?ucs_event_no=&bo_table=ucs&sca=&sop=and&sfl=wr_name%2C1&stx='
UCS_SONG_URL = 'http://www.piugame.com/bbs/board.php?ucs_event_no=&bo_table=ucs&sca=&sop=and&sfl=ucs_song_no%2C1&stx='
BASE_DL_URL = 'http://piugame.com/bbs/'
BASE_LOGIN_URL = 'https://www.piugame.com/bbs/piu.login_check.php'

OUT_DIR = os.path.join(str(ABS_PATH), '../dataset/raw/')

UCS_LVL_PATTERN = 's_lv([0-9][0-9])'
UCS_CODE_PATTERN = '_(CS[0-9][0-9][0-9])#scrollSpeed='

class UCS_Spider(scrapy.Spider):
    name = 'ucs_spider'

    # specific > generic (step_artist > song > level > mix > ...)
    def __init__(self, pack_name, step_artists, min_level, max_level,
        chart_type, songs, start_date, end_date, meta_json_path):

        self.pack_name = pack_name

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
            start_urls = [BASE_LOGIN_URL]
        
        # login to ucs site to enable downloads using 'requests'
        # https://stackoverflow.com/questions/11892729/how-to-log-in-to-a-website-using-pythons-requests-module/17633072#17633072
        print('Please enter login credentials: ')
        user_email = input('User Email: ')
        user_pw = getpass('Password: ')

        payload = {'mb_id': user_email, 'mb_password': user_pw}

        self.session_requests = requests.session()
        result = self.session_requests.post(BASE_LOGIN_URL, data=payload,
            headers={'referer': BASE_LOGIN_URL})

    def parse(self, response):
        result = self.session_requests.get(response.url, headers={'referer': response.url})
        new_response = html.fromstring(result.content)

        ucs_rows = new_response.xpath('//div[@class="share_list"]/table/tbody/tr')

        for ucs_row in ucs_rows:
            self.parse_ucs_row(ucs_row)
            
        page_area = new_response.xpath('//span[@class="pg"]')[0]

        curr_page = int(page_area.xpath('.//strong[@class="pg_current"]/text()')[0])
        next_page = curr_page + 1
        next_page_link = page_area.xpath('.//a[contains(@href, "page=' + str(next_page) + '")]/@href')

        if next_page_link:
            yield response.follow(next_page_link[0], callback=self.parse)

    def parse_ucs_row(self, response):
        # date in (YY-MM-DD) format
        upload_date = response.xpath('.//td[@class="share_upload_date"]/text()')[0].split('-')
        upload_date_obj = datetime.date(year=int('20' + upload_date[0]),
            month=int(upload_date[1]), day=int(upload_date[2]))
        correct_date = self.start_date < upload_date_obj and upload_date_obj < self.end_date

        if not correct_date:
            return

        song_title = response.xpath('.//span[@class="share_song_title"]/a/text()')[0].strip().lower()

        if not self.songs or self.songs and song_title in self.songs:
            chart_leveltype = response.xpath('.//td[@class="share_level"]/span/@class')[0]
            chart_level = int(re.search(UCS_LVL_PATTERN, chart_leveltype).group(1))

            correct_chart_type = self.chart_type in chart_leveltype 
            valid_level = self.min_level <= chart_level and chart_level <= self.max_level

            if correct_chart_type and valid_level:
                stepmaker = response.xpath('.//td[@class="share_stepmaker"]/text()')[0].strip().lower()
                ucs_code = re.search(UCS_CODE_PATTERN, tostring(response, encoding='unicode')).group(1)
                
                dl_link = response.xpath('.//a[@class="share_download2"]/@href')[0]

                this_ucs_metadata = self.UCS_METADATA[ucs_code]

                ucs_dl_dict = {**this_ucs_metadata, **{'url': BASE_DL_URL + dl_link, 
                    'name': ucs_code, 'step_artist': stepmaker, 
                    'chart_type': self.chart_type, 'meter': chart_level,
                    'songtype': 'arcade', 'pack_name': self.pack_name}}

                self.download_chart(ucs_dl_dict) 
                    
    # download the specified ucs chart with its associated data
    def download_chart(self, ucs_dict):
        chart_txt = self.session_requests.get(ucs_dict['url']).text

        # create + download to subfolder
        clean_title = ez_name(ucs_dict['title'])
        clean_packname = ez_name(ucs_dict['pack_name'])
        clean_artist = ez_name(ucs_dict['step_artist'])
        
        ucs_folder_name = ucs_dict['chart_type'] + str(ucs_dict['meter']) + '_' + clean_artist
        
        ucs_dir = os.path.join(OUT_DIR, clean_packname, clean_title + '_' + ucs_folder_name)
        if not os.path.isdir(ucs_dir):
            os.makedirs(ucs_dir)

        self.add_chart_data(chart_txt, ucs_dict, ucs_dir)

    # write the chart data to the specified file
    def add_chart_data(self, chart_txt, ucs_dict, dir_fp):
        ucs_text = ''
        for k,v in ucs_dict.items():
            ucs_text += ':{}={}\n'.format(k,v)

        ucs_text += chart_txt

        ucs_fp = os.path.join(dir_fp, ucs_dict['name'] + '.ucs')
        with open(ucs_fp, 'w') as f:
            f.write(ucs_text)