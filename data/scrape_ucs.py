# scrape UCS charts

import os
import scrapy
import pathlib

UCS_URL = 'http://www.piugame.com/bbs/board.php?bo_table=ucs'

ABS_PATH = pathlib.Path(__file__).parent.absolute()
OUT_PATH = os.path.join(str(ABS_PATH), 'dataset/raw/UCS')



def main():
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)   



if __name__ == '__main__':
    main()