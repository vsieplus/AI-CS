# train a pair of models on the same dataset to perform step placement/selection

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import stepchart
from placement_model import PlacementCNN, PlacementRNN

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

BATCH_SIZE = 1

PLACEMENT_CRITERION = nn.BCELoss()
PLACEMENT_EPOCHS = 6
PLACEMENT_LR = 0.005

SELECTION_CRITERION = nn.CrossEntropyLoss()
SELECTION_EPOCHS = 6
SELECTION_LR = 0.005

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, 
        help='Name of dataset under data/datasets/subsets/')
    parser.add_argument('--dataset_dir', type=str, default=None, 
        help='Alternatively, provide direct path to dataset')
    parser.add_argument('--output_dir', type=str, default=None, help="""Specify
        custom output directory to save models to. If blank, will save in
        ./models/dataset_name""")

    args = parser.parse_args()

    if not args.dataset_dir and args.dataset_name:
        args.dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name)
    
    if not args.output_dir:
        args.output_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_dir)[-1])

    return args

# train the placement models with the specified parameters
def train_placement_batch(cnn, rnn, optimizer, criterion, input):

    # bptt https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/29
    # unrolling https://machinelearningmastery.com/rnn-unrolling/


    cnn.train()
    rnn.train()

def train_selection_batch(rnn, optimizer, criterion, input):
    pass

# train on a single batch of examples
def train_batch(batch, early_stopping=True):
    pass

# full training process from placement -> selection
def train(num_epochs, batch_size, early_stopping=True):
    pass

def main():
    args = parse_args()

    # Retrieve/prepare data

    
    # train_size = int(0.8 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_data, valid_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create + train placement model
    cnn = PlacementCNN()
    rnn = PlacementRNN(num_lstm_layers=2, num_features=30)

    placement_optim = optim.Adam(list(cnn.parameters()) + list(rnn.parameters()), lr=PLACEMENT_LR)


    # Train selection model


if __name__ == '__main__':
    main()