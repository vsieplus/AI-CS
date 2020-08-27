# Use a set of pretrained models to generate a step chart

import argparse
import json
import os

import torch

import sys
sys.path.append(os.path.join('..', 'train'))
from hyper import N_CHART_TYPES, N_LEVELS
from arrow_rnns import PlacementCLSTM, SelectionRNN
from arrow_transformer ArrowTransformer
import extract_audio_feats
import train_rnns

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='path to directory containing model files')
    parser.add_argument('--audio_file', type=str, help='path to audio file to use')
    parser.add_argument('--chart_format', type=str, default='ucs', choices=['ucs', 'ssc', 'both'] help='output format to use')
    parser.add_argument('--out_dir', type=str, required=True, help='where to save output files')
    parser.add_argument('--level', type=int, default=14, help='level of the chart to generate')
    parser.add_argument('--song_name', type=str, help='song name')
    parser.add_argument('--song_artist', type=str, help='song artist')
    parser.add_argument('--bpm', type=float, default=140, help='provide true song bpm if known')
    parser.add_argument('--sampling', type=str, default='top-p', choices=['top-p', 'top-k', 'greedy', 'multinom'], 
                        help='choose the sampling strategy to use when generating the step sequence')
    parser.add_argument('-k', type=int, default=100, 'the k to use in top-k sampling')
    parser.add_argument('-p', type=int, default=0.9, 'the p to use in top-p sampling')

    return parser.parse_args()

def save_chart(chart_data, chart_format, song_name, artist, bpm, special_tokens, out_dir):
    if not os.path.isdir(out_dir):
        print(f'Creating output directory {out_dir}')
        os.makedirs(out_dir)

    # Obtain ucs representation first, then convert if needed
    


    if chart_format == 'ssc' or if chart_format == 'both':
        pass

def generate_chart(placement_model, selection_model, audio_file, chart_type, 
                   chart_level, n_step_features, sampling, k, p, device):
    placement_model.eval()
    selection_model.eval()

    print('Generating chart - this may take a bit...')

    # store pairs of time (s) and step vocab indices
    chart_data = []

    # shape [3, ?, 80] -> [batch=1, 3, ?, 80]
    audio_feats = extract_audio_feats.extract_audio_feats(audio_file).unsqueeze(0)
    n_audio_frames = audio_feats.size(2)

    chart_feats = [0] * (N_CHART_TYPES + N_LEVELS)

    if chart_type == 'pump-single':
        chart_feats[0] = 1
    else:
        chart_feats[1] = 1

    chart_feats[chart_level + 1] = 1

    # [batch=1, # chart features]
    chart_feats = torch.tensor(chart_feats, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        states = placement_model.rnn.initStates(batch_size=1, device=device)
        audio_length = torch.tensor([n_audio_frames], dtype=torch.long, device=device)

        # [batch=1, n_audio_frames, 2] / [batch=1, n_audio_frames, hidden] / ...
        logits, placement_hiddens, _ = clstm(audio_feats, chart_feats, states, audio_length)

        # placement predictions - [n_audio_frames] - 1 if a placement, 0 if empty
        placements = train_rnns.predict_placements(logits, [chart_level], audio_length).squeeze(0)

        placement_frames = (placements == 1).nonzero(as_tuple=False).flatten()
        num_placements = placement_frames.sum().item()

        # Start generating the sequence of steps
        step_length = torch.ones(1, dtype=torch.long, device=device)
        
        for i in range(num_placements):
            placement_melframe = placement_frames[i]
            # placement_time = 

            if i == 0:
                start_token = torch.zeros(1, n_step_features, device=device)
                hidden, cell = selection_model.initStates(batch_size=1, device=device)
                logits, (hidden, cell) = selection_model(start_token, None, hidden, cell, step_length)
            else:
                placement_hidden = placement_hiddens[0, placement_melframe]
                logits, (hidden, cell) = selection_model(next_token, placement_hidden, hidden, cell, step_length)

            next_step = predict_step(logits, sampling, k, p)


def predict_step(logits, sampling, k, p):
    """predict the next step given model logits and a sampling strategy"""
    dist = torch.nn.functional.softmax(logits, dim=-1)
    
    if sampling == 'top-k':
        # sample from top k probabilites + corresponding indices (sorted)    
        probs, indices = torch.topk(dist, k=k, dim=-1)
        sample = torch.multinomial(torch.nn.functional.softmax(probs, dim=-1), num_samples=1)

        pred_idx = indices[sample]
    elif sampling == 'top-p':
        # sample from smallest set that cumulatively exceeds probability 'p'
        # https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

        filtered_dist = torch.nn.functional.softmax(logits, dim=-1)
        pred_idx = torch.multinomial(filtered_dist, num_samples=1)

    elif sampling == 'greedy':
        # take the most likely token
        pred_idx = torch.topk(dist, k=1, dim=-1)[1][0]
    elif sampling == 'multinom':
        # sample from the bare distribution
        pred_idx = torch.multinomial(dist, num_samples=1)

    return pred_idx

def main():
    args = parse_args()

    device = torch.device('cpu')

    with open(os.path.join(args.model_dir, 'summary.json'), 'r') as f:
        model_summary = json.loads(f.read())

    if model_summary['type'] == 'rnns':
        placement_model = PlacementCLSTM(model_summary['placement_channels'], model_summary['placement_filters'], 
                                         model_summary['placement_kernels'], model_summary['placement_pool_kernel'],
                                         model_summary['placement_pool_stride'], model_summary['placement_lstm_layers'],
                                         model_summary['placement_input_size'], model_summary['hidden_size']).to(device)

        selection_model = SelectionRNN(model_summary['selection_lstm_layers'], model_summary['selection_input_size'], 
                                       model_summary['vocab_size'], model_summary['hidden_size'],
                                       model_summary['selection_hidden_wt']).to(device)

        # loads the state dicts from the .bin files in model_dir/
        train_rnns.load_save(args.model_dir, False, placement_model, selection_model, device)
    elif model_summary['type'] == 'transformer':
        pass

    # a list of pairs of (absolute time (s), steps vocab index)
    chart_data = generate_chart(placement_model, selection_model, args.audio_file, model_summary['chart_type'],
                                args.level, model_summary['selection_input_size'], args.sampling, k, p, device)

    if os.path.isfile(os.path.join(args.model_dir, 'special_tokens.json')):
        with open(os.path.join(args.model_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.loads(f.read())
    else:
        special_tokens = None

    # convert indices -> chart output format + save to file
    save_chart(chart_data, args.chart_format, args.song_name, args.song_artist, 
               args.bpm, special_tokens, args.out_dir)

if __name__ == '__main__':
    main()