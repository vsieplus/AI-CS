#!/bin/bash

# script to download game song/chart data for training 
# (packs in StepF2/P1 (+infinity) as of 7/18/2020)

URL='https://drive.google.com/uc?id='
ZIP_OUTFILE='PIU_PACKS.zip'
DATASET_DIR='dataset/'

gdown ${URL} -O ${ZIP_OUTFILE}
tar xvf ${ZIP_OUTFILE} -C  ${DATASET_DIR}
rm ${ZIP_OUTFILE}