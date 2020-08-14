# train a pair of models on the same dataset to perform step placement/selection

import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hyper import *
from arrow_rnns import PlacementCNN, PlacementRNN, SelectionRNN
from arrow_seq2seq import AudioEncoder, ArrowDecoder
from arrow_transformer import ArrowTransformer
from stepchart import StepchartDataset, get_splits, collate_charts
from util import report_memory

ABS_PATH = str(Path(__file__).parent.absolute())
DATASETS_DIR = os.path.join(ABS_PATH, '../data/dataset/subsets')
MODELS_DIR = os.path.join(ABS_PATH, 'models')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of dataset under data/datasets/subsets/')
    parser.add_argument('--dataset_path', type=str, default=None, 
        help='Alternatively, provide direct path to dataset json file')
    parser.add_argument('--save_dir', type=str, default=None, help="""Specify custom output directory to save
        models to. If blank, will save in models/dataset_name""")
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Load models from the specified checkpoint')
    parser.add_argument('--restart', action='store_true', default=False,
        help='Use this option to start training from beginning with a checkpoint')

    args = parser.parse_args()

    if not args.dataset_path:
        if args.dataset_name:
            args.dataset_path = os.path.join(DATASETS_DIR, args.dataset_name + '.json')
        else:
            raise ValueError('--dataset_name or --dataset_path required')

    if not args.save_dir:
        args.save_dir = os.path.join(MODELS_DIR, os.path.split(args.dataset_path)[-1])

    return args

def save_checkpoint(p_cnn, p_rnn, s_rnn, p_optim, s_optim, epoch, curr_batch,
                    best_p_vloss, best_s_vloss, save_dir, best=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    out_path = os.path.join(save_dir, 'checkpoint.tar')
    
    print(f'Saving checkpoint to {out_path}')
    torch.save({
        'epoch': epoch,
        'curr_batch': curr_batch,
        'best_placement_validation_loss': best_p_vloss,
        'best_selection_validation_loss': best_s_vloss,
        'p_cnn_state_dict': p_cnn.state_dict(),
        'p_rnn_state_dict': p_rnn.state_dict(),
        'p_optimizer': p_optim.state_dict(),
        's_rnn_state_dict': s_nn.state_dict(),
        's_optimizer': s_optim.state_dict()
    }, out_path)

    if best:
        shutil.copy(out_path, 'model_best.tar')

# train the placement models with the specified parameters on the given batch
def train_placement_batch(cnn, rnn, placement_params, optimizer, criterion, batch, device):
    cnn.train()
    rnn.train()

    # [batch, 3, ?, 80] ; ? = (batch maximum) # of 10ms timesteps/frames in audio features
    audio_feats = batch['audio_feats'].to(device)
    num_audio_frames = audio_feats.size(2)
    batch_size = audio_feats.size(0)

    # [batch, # chart features]
    chart_feats = batch['chart_feats'].to(device)

    # both are [batch, max # of chart frames] (padded/not padded)
    chart_placements = batch['placement_targets']   # if chart frame had a hit or not
    step_frames = batch['step_frames']              # which audio frames recorded in chart data

    # the indices of step_frames which had step placements
    step_placement_indices = [(chart_placements[b] == 1).nonzero(as_tuple=False).squeeze().tolist()
                               for b in range(batch_size)]

    # the subset of step_frames (audio frames) that had placements
    step_placement_frames = torch.tensor([[step_frames[b][idx] for idx in step_placement_indices[b]]
                                           for b in range(batch_size)])

    # compute the first and last audio frames coinciding with chart non-emtpy step placements
    first_frame = min(step_frames[b][indices[0]] for b, indices in enumerate(step_placement_indices))
    last_frame = max(step_frames[b][indices[-1]] for b, indices in enumerate(step_placement_indices))

    total_loss = 0

    # final_shape -> [batch, # of nonempty chart frames, hidden]
    clstm_hiddens = [[] for _ in range(batch_size)]

    # unrolling/bptt
    num_unrollings = (num_audio_frames + 1) // PLACEMENT_UNROLLING_LEN
    last_unroll_len = num_audio_frames % PLACEMENT_UNROLLING_LEN
    states = rnn.initStates(batch_size, device)
    
    # get representation of audio lengths for each unrolling
    # [full frames, last frame length]
    # (e.g. [100, 100, 100, 64] -> (3,64))
    audio_unroll_lengths = [(batch['audio_lengths'][b] // PLACEMENT_UNROLLING_LEN, 
                             batch['audio_lengths'][b] % PLACEMENT_UNROLLING_LEN) 
                            for b in range(batch_size)]

    # loop through all unrollings
    for unrolling in range(num_unrollings):
        audio_start_frame = unrolling * PLACEMENT_UNROLLING_LEN
        if unrolling == num_unrollings - 1:
            audio_end_frame = audio_start_frame + last_unroll_len
        else:
            audio_end_frame = audio_start_frame + PLACEMENT_UNROLLING_LEN
        unroll_length = audio_end_frame - audio_start_frame

        # skip audio frames outside chart range
        if (audio_start_frame + unroll_length < first_frame or 
            audio_end_frame - unroll_length > last_frame + 1 or unroll_length == 0):
            continue

        # compute relevant batch metavalues for padding/lengths
        padded_seqs = [unrolling > audio_unroll_lengths[b][0] - 1 for b in range(batch_size)]
        audio_lengths = [audio_unroll_lengths[b][1] if padded_seqs[b] else unroll_length
                         for b in range(batch_size)]

        # [batch, unroll_len, 83] (batch, unroll length, audio features)
        cnn_input = audio_feats[:, :, audio_start_frame:audio_end_frame]
        cnn_output = cnn(cnn_input)

        # [batch, unroll_len, 2] / [batch, unroll_len, hidden] / (*...hidden x 2)
        logits, clstm_hidden, states = rnn(cnn_output, chart_feats, states, audio_lengths)

        # for each batch example, if this audio frame also in the chart, use actual target at that frame
        # otherwise use default zero value for no placement at all
        targets = torch.zeros(batch_size, unroll_length, dtype=torch.long, device=device)
        for b in range(batch_size):
            # which frames to consider for the current unrolling
            curr_unroll_frames = torch.logical_and(step_placement_frames[b] >= audio_start_frame,
                                                   step_placement_frames[b] < audio_end_frame)

            # which of them had placements?
            curr_unroll_placements = step_placement_frames[b][curr_unroll_frames]

            # (normalize to [0, unroll_length - 1])
            curr_unroll_placements = curr_unroll_placements - audio_start_frame

            targets[b, curr_unroll_placements] = 1

            # for padded seqs, set targets to the ignore_index val.
            targets[b, audio_lengths[b]:] = PAD_IDX

            # only clstm hiddens for chart frames with non-empty step placements
            clstm_hiddens[b].append(clstm_hidden[b, curr_unroll_placements])

        # compute total loss for this unrolling
        loss = 0
        for step in range(unroll_length):
            loss += criterion(logits[:, step, :], targets[:, step])

        loss.backward()
        
        # clip grads before step if L2 norm > 5
        nn.utils.clip_grad_norm_(placement_params, max_norm=PLACEMENT_MAX_NORM)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss_per_timestep = total_loss / num_audio_frames
    
    return avg_loss_per_timestep, clstm_hiddens

def train_selection_batch(rnn, optimizer, criterion, batch, device, clstm_hiddens=None):
    rnn.train()

def evaluate(p_cnn, p_rnn, s_rnn, valid_iter, p_criterion, s_criterion, device):
    s_rnn.eval()

    total_placement_loss = 0

    with torch.no_grad():
        for batch in valid_iter:
            total_placement_loss += evaluate_placement_batch(p_cnn, p_rnn, batch)

    return total_loss

def evaluate_placement_batch(cnn, rnn, criterion, batch):
    cnn.eval()
    rnn.eval()


def evaluate_selection_batch(rnn, criterion, batch, clstm_hiddens):
    rnn.eval()

# full training process from placement -> selection
def run_models(train_iter, valid_iter, num_epochs, dataset_type, device, save_dir, load_checkpoint,
               restart, early_stopping=True, print_every_x_batch=1, validate_every_x_epoch=2, 
               save_every_x_batch=8):
    cpu_device = torch.device('cpu')

    # setup  or load models, optimizers
    placement_cnn = PlacementCNN(PLACEMENT_CHANNELS, PLACEMENT_FILTERS, PLACEMENT_KERNEL_SIZES,
                                 PLACEMENT_POOL_KERNEL, PLACEMENT_POOL_STRIDE, device).to(device)
    placement_rnn = PlacementRNN(NUM_PLACEMENT_LSTM_LAYERS, PLACEMENT_INPUT_SIZE, HIDDEN_SIZE).to(device)
    placement_params = list(placement_cnn.parameters()) + list(placement_rnn.parameters())
    placement_optim = optim.SGD(placement_params, lr=PLACEMENT_LR, momentum=0.9)

    selection_rnn = SelectionRNN(NUM_SELECTION_LSTM_LAYERS, SELECTION_VOCAB_SIZES[dataset_type], 
                                 HIDDEN_SIZE, SELECTION_HIDDEN_WEIGHT).to(device)
    selection_optim = optim.SGD(selection_rnn.parameters(), lr=SELECTION_LR, momentum=0.9)

    # load model, optimizer states if resuming training
    if load_checkpoint:
        if os.path.isfile(load_checkpoint):
            print(f'Loading checkpoint from {load_checkpoint}...')
            checkpoint = torch.load(load_checkpoint)

            # only restore epoch/best loss values when not restarting    
            # (i.e. give option to restart training on a pretrained-model)       
            if not restart:
                start_epoch = checkpoint['epoch']
                curr_batch = checkpoint['curr_batch']
                best_placement_valid_loss = checkpoint['best_placement_validation_loss']
                best_selection_valid_loss = checkpoint['best_selection_validation_loss']

            placement_cnn.load_state_dict(checkpoint['p_cnn_state_dict'])
            placement_rnn.load_state_dict(checkpoint['p_rnn_state_dict'])
            selection_rnn.load_state_dict(checkpoint['s_rnn_state_dict'])

            placement_optim.load_state_dict(checkpoint['p_optimizer'])
            selection_optim.load_state_dict(checkpoint['s_optimizer'])
        else:
            print(f'Invalid path to checkpoint: {load_checkpoint}')
            return
    else:
        best_placement_valid_loss = float('inf')
        best_selection_valid_loss = float('inf')
        start_epoch = 0
        curr_batch = 0

    print('Starting training..')
    for epoch in tqdm(range(num_epochs)):
        if epoch < start_epoch:
            continue

        print('Epoch: {}'.format(epoch))
        # if device is gpu: report_memory()

        for i, batch in enumerate(train_iter):
            placement_loss, clstm_hiddens = train_placement_batch(placement_cnn,
                placement_rnn, placement_params, placement_optim, PLACEMENT_CRITERION, batch, device)

            #selection_loss = train_selection_batch(selection_rnn, selection_optim,
            #    SELECTION_CRITERION, batch, device, clstm_hiddens)

            if (i + 1) % print_every_x_batch == 0:
                print(f'Batch {i}')
                print(f'Placement model batch {i} avg. loss/timestep: {placement_loss}')        
                #print(f'Selection model batch loss: {selection_loss}')       

            if (curr_batch + 1) % save_every_x_batch == 0:
                save_checkpoint(placement_cnn, placement_rnn, selection_rnn,
                                placement_optim, selection_optim, epoch, curr_batch,
                                best_placement_valid_loss, best_selection_valid_loss, save_dir)

            curr_batch += 1

        if (epoch + 1) % validate_every_x_epoch == 0:
            placement_valid_loss = evaluate(placement_cnn, placement_rnn, selection_rnn,
                valid_iter, PLACEMENT_CRITERION, SELECTION_CRITERION, device)

            # track best performing model
            if placement_valid_loss < best_placement_valid_loss:
                best_placement_valid_loss = placement_valid_loss
                save_checkpoint(placement_cnn, placement_rnn, selection_rnn, save_dir,
                    placement_optim, selection_optim, epoch, curr_batch,
                    best_placement_valid_loss, best_selection_valid_loss, best=True)
            else:
                break # early stopping

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)

    # Retrieve/prepare data
    print('Loading dataset from {}...'.format(os.path.relpath(args.dataset_path)))
    dataset = StepchartDataset(args.dataset_path)
    dataset_type = dataset.chart_type

    train_data, valid_data, test_data = get_splits(dataset)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)
    valid_iter = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_charts)

    print(f"""Total charts in dataset: {len(dataset)}; Train: {len(train_data)}, 
        Valid: {len(valid_data)}, Test: {len(test_data)}""")

    run_models(train_iter, valid_iter, NUM_EPOCHS, dataset_type, device, args.save_dir, 
        args.load_checkpoint, args.restart)

if __name__ == '__main__':
    main()
