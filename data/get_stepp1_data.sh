#!/bin/bash

# script to download game song/chart data for training 
# (packs in StepF2/P1 (+infinity) as of 7/18/2020) [ucs/mp3 only]
URL='https://drive.google.com/uc?id=1a_NnUohVzyp4vNwON9UJkbkzL-mMujwk'
ZIP_OUTFILE='PIU_PACKS.zip'
DATASET_DIR='dataset/raw/'

mkdir -p ${DATASET_DIR}

gdown ${URL} -O ${ZIP_OUTFILE}

echo "unzipping ${ZIP_OUTFILE}"
unzip -d ${DATASET_DIR} ${ZIP_OUTFILE}