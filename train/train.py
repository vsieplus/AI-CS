# train a pair of models on the same dataset to perform step placement/selection

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from stepchart import StepchartDataset, get_splits
from placement_model import PlacementCNN, PlacementRNN
from selection_model import SelectionRNN

ABS_PATH = str(pathlib.Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

BATCH_SIZE = 1
NUM_EPOCHS = 5

PLACEMENT_CRITERION = nn.BCEWithLogitsLoss()
PLACEMENT_LR = 0.005

PLACEMENT_CHANNELS = [3, 10]
PLACEMENT_FILTERS = [10, 20]
PLACEMENT_KERNEL_SIZES = [(7, 3), (3, 3)]

PLACEMENT_HIDDEN_SIZE = 256
NUM_PLACEMENT_LSTM_LAYERS = 2
NUM_PLACEMENT_FEATURES = 30

SELECTION_CRITERION = nn.CrossEntropyLoss()
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
def train_placement_batch(cnn, rnn, optimizer, batch):

    # bptt https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/29
    # unrolling https://machinelearningmastery.com/rnn-unrolling/

    cnn.train()
    rnn.train()



    chart_types = batch.chart_type
    chart_levels = batch.level

#    chart_feats_onehot = torch.zeros(batch_size, num_feats).scatter(1, ..., 1)
    
def train_selection_batch(rnn, optimizer, batch, lstm_hiddens=None):
    rnn.train()

def evaluate_placement_batch(cnn, rnn, batch):
    cnn.eval()
    rnn.eval()

def evaluate_selection_batch(rnn, batch):
    rnn.eval()

# full training process from placement -> selection
def train(train_iter, valid_iter, num_epochs, device, early_stopping=True,
    print_every_x_batch=10, validate_every_x_epoch=3):

    # setup models, optimizers
    placement_cnn = PlacementCNN(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES)
    placement_rnn = PlacementRNN(NUM_PLACEMENT_LSTM_LAYERS, NUM_PLACEMENT_FEATURES, PLACEMENT_HIDDEN_SIZE)
    placement_optim = optim.Adam(list(cnn.parameters()) + list(rnn.parameters()), lr=PLACEMENT_LR)

    selection_rnn = SelectionRNN(pass)
    selection_optim = optim.Adam(selection_rnn.parameters(), lr=SELECTION_LR)

    print('Starting training..')
    for epoch in range(num_epochs):
        print('Epoch: {}'.fomrat(epoch))

        for i, batch in enumerate(train_iter):
            placement_batch = batch['song']['...']

            avg_placement_loss, lstm_hiddens = train_placement_batch(placement_cnn,
                placement_rnn, placement_optim, batch)

            selection_batch = batch['steps']['..']
            avg_selection_loss = train_selection_batch(selection_rnn, selection_optim,
                selection_batch, lstm_hiddens)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Retrieve/prepare data
    print('Loading dataset from {}...'.format(os.path.relpath(args.dataset_dir)))
    dataset = StepchartDataset(args.dataset_dir)
    
    train_data, valid_data, _ = get_splits(dataset)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)

    train(train_iter, valid_iter, NUM_EPOCHS, device)

if __name__ == '__main__':
    main()