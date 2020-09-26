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
from hyper import (N_MELS, N_FFTS, SELECTION_VCOAB_SIZES, SELECTION_INPUT_SIZES, PLACEMENT_LR, SELECTION_LR,
                   PLACEMENT_CRITERION, SELECTION_CRITERIOn)
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

def run_placement_batch():
    pass

def run_selection_batch():
    pass

def evaluate():
    pass

def run_training(train_iter, valid_iter, test_iter, chart_type, save_dir,
                 placement_modelname, selection_modelname, device, num_epochs=15):
    # flattened audio feats as input
    placement_input_size = N_MELS * len(N_FFTS)
    if placement_modelname == 'logreg':
        placement_model = baseline.PlacementLogReg(input_size=placement_input_size, output_size=2)
    elif placement_modelname == 'mlp':
        placement_model = baseline.PlacementMLP(input_size=placement_input_size, output_size=2)

    if selection_modelname == 'ngram':
        ngram_counts = Counter()
        for json_fp, _, _, _ in train_iter.dataset.chart_ids:
            with open(json_fp, 'r') as f:
			    attrs = json.loads(f.read())
            for chart_attrs in attrs['charts']:
                notes = [note for _, _, _, note in chart_attrs['notes']]
                for ngram in get_ngrams(notes):
                    ngram_counts[ngram] += 1        
        selection_model = baseline.SelectionNGram(ngram_counts)
    elif selection_modelname == 'ngrammlp':
        selection_model = baseline.SelectionNGramMLP(SELECTION_VCOAB_SIZES[chart_type],
                                                     SELECTION_INPUT_SIZES[chart_type])

    placement_optim = optim.SGD(placement_model.parameters(), lr=PLACEMENT_LR)
    selection_optim = optim.SGD(selection_model.parameters(), lr=SELECTION_LR)

    placement_save = MODEL_SAVENAMES[placement_modelname]
    selection_save = MODEL_SAVENAMES[selection_modelname]

    best_placement_valid_loss = float('inf')
    best_placement_precision = 0
    best_selection_valid_loss = float('inf')
    train_placement, train_selection = True, True

    print('Starting training..')
    for epoch in trange(num_epochs):
        print('Epoch: {}'.format(epoch))
        epoch_p_loss = 0
        epoch_p_precision = 0
        epoch_s_loss = 0

        for i, batch in enumerate(tqdm(train_iter)):
            with torch.set_grad_enabled(train_placement):
                (placement_loss,
                 placement_acc,
                 placement_precision) = run_placement_batch(placement_model, placement_optim, PLACEMENT_CRITERION, 
                                                            batch, device, do_train=train_placement)

            with torch.set_grad_enabled(train_selection):
                (selection_loss,
                 selection_acc) = run_selection_batch(selection_model, selection_optim, SELECTION_CRITERION,
                                                      batch, device, do_train=train_selection)

            epoch_p_loss += placement_loss
            epoch_p_precision += placement_precision
            epoch_s_loss += selection_loss

            if train_placement:
                save_model(placement_model, save_dir, placement_save)
            if train_selection:
                save_model(selection_model, save_dir, selection_save)

        epoch_p_loss = epoch_p_loss / len(train_iter)
        epoch_p_precision = epoch_p_precision / len(train_iter)
        epoch_s_loss = epoch_s_loss / len(train_iter)

        print(f'\tAvg. training placement loss per unrolling: {epoch_p_loss:.5f}')
        print(f'\tAvg. training placement precision: {epoch_p_precision:.5f}')
        print(f'\tAvg. training selection loss per frame: {epoch_s_loss:.5f}')

        (placement_valid_loss,
         placement_valid_acc,
         selection_valid_loss,
         selection_valid_acc,
         placement_precision) = evaluate(placement_model, selection_model, valid_iter, 
                                        PLACEMENT_CRITERION, SELECTION_CRITERION, device)
            
        print(f'\tAvg. validation placement loss per frame: {placement_valid_loss:.5f}')
        print(f'\tAvg. training placement precision: {placement_precision:.5f}')
        print(f'\tAvg. validation selection loss per frame: {selection_valid_loss:.5f}')

        better_placement = placement_precision > best_placement_precision
        better_selection = selection_valid_loss < best_selection_valid_loss

        if train_placement:
            if better_placement:
                best_placement_precision = placement_precision				
                best_placement_valid_loss = placement_valid_loss
                save_model(placement_clstm, save_dir, placement_save)
            else:
                print("Placement validation loss increased, stopping CLSTM training")
                train_placement = False

        if train_selection:
            if better_selection:
                best_selection_valid_loss = selection_valid_loss
                save_model(selection_rnn, save_dir, selection_save)
            else:
                print("Placement validation loss increased, stopping SRNN training")
                train_selection = False

        if not train_placement and not train_selection:
            print("Both early stopping criterion met. Stopping early..")
            break

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

    run_training(train_iter, valid_iter, test_iter, dataset.chart_type, save_dir,
                 args.baseline_placement, args.baseline_selection, device)

if __name__ == '__main__':
    main()