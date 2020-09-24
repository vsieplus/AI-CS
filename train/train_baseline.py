# training script for baseline models

import argparse
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score

import baseline_models
from stepchart import StepchartDataset, collate_charts
from train_util import report_memory, SummaryWriter, load_save, save_checkpoint, save_model, get_dataloader

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

def get_ngrams(n, notes, prepend=True, append=True):
    prepended = []
    if prepend:
        prepend = [f'<pre{i}>' for i in reversed(range(n - 1))]

    appended = ['<post>'] if append else []

    sequence = prepended + notes + appended
    for i in range(len(sequence) - n + 1):
        yield tuple(sequence[i:i + n])

def main():
    parser = argparse.ArgumentParser()

if __name__ == '__main__':
    main()