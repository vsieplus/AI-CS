# training script for baseline models

import argparse
import os
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score

import baseline_models as baseline
from hyper import N_MELS, N_FFTS
from stepchart import StepchartDataset, collate_charts
from train_util import report_memory, SummaryWriter, load_save, save_checkpoint, save_model, get_dataloader

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models', 'baseline')

MODEL_SAVENAMES = {
    'logreg': 'placement_logreg.bin',
    'mlp': 'placement_mlp.bin',
    'ngram': 'selection_ngram.bin',
    'ngrammlp': 'selection_ngram_mlp.bin'
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset under ../data/datasets/subsets/')
    parser.add_argument('--baseline_placement', type=str, default='logreg', choices=['logreg', 'mlp'], help='which placement model to run')
    parser.add_argument('--baseline_selection', type=str, default='ngram', choices=['ngram', 'ngrammlp'], help='which selection model to run')
    args = parser.parse_args()

    return args

def get_ngrams(notes, n=5, prepend=True, append=True):
    prepended = []
    if prepend:
        prepend = [f'<pre{i}>' for i in reversed(range(n - 1))]

    appended = ['<post>'] if append else []

    sequence = prepended + notes + appended
    for i in range(len(sequence) - n + 1):
        yield tuple(sequence[i:i + n])

def run_training(train_iter, valid_iter, save_dir, placement_modelname, selection_modelname):
    # flattened audio feats as input
    placement_input_size = N_MELS * len(N_FFTS)
    if placement_modelname == 'logreg':
        placement_model = baseline.PlacementLogReg(input_size=placement_input_size, output_size=2)
    elif placement_modelname == 'mlp':
        placement_model = baseline.PlacementMLP(input_size=placement_input_size, output_size=2)

    if selection_modelname == 'ngram':
        ngram_counts = Counter()
        for chart_fp, _, _, _ in train_iter.dataset.chart_ids:
            

        
        selection_model = baseline.SelectionNGram(ngram_counts)
    elif selection_modelname == 'ngrammlp':

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Retrieve/prepare data
    dataset_path = os.path.join(DATASETS_DIR, args.dataset_name + '.json')
    print(f'Loading dataset from {dataset_path}...')
    dataset = StepchartDataset(dataset_path, load_to_memory=False, first_dataset_load=True, special_tokens=None)

    # models/baseline/{single/double}/dataset_name/...
    save_dir = os.path.join(MODELS_DIR, dataset.chart_type.split('-')[-1],
                            os.path.split(dataset_path)[-1].split('.')[0])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_indices, valid_indices, test_indices = dataset.get_splits()
    train_iter = get_dataloader(dataset, BATCH_SIZE, train_indices)
    valid_iter = get_dataloader(dataset, BATCH_SIZE, valid_indices)
    test_iter = get_dataloader(dataset, BATCH_SIZE, test_indices)

    run_training(train_iter, valid_iter, test_iter, save_dir, args.baseline_placement,
                 args.baseline_selection)

if __name__ == '__main__':
    main()