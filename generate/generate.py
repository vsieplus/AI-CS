# Use a set of pretrained models to generate a step chart

import argparse
import json
import os
import shutil

import torch
from tqdm import trange

import sys
sys.path.append(os.path.join('..', 'train'))
from hyper import N_CHART_TYPES, N_LEVELS, CHART_FRAME_RATE
from arrow_rnns import PlacementCLSTM, SelectionRNN
from arrow_transformer import ArrowTransformer
from stepchart import step_features_to_str, step_index_to_features, UCS_SSC_DICT
from extract_audio_feats import extract_audio_feats
import train_rnns
import util

BEATS_PER_MEASURE = 4
SPLITS_PER_BEAT = 32
SPLIT_SUBDIV = SPLITS_PER_BEAT * BEATS_PER_MEASURE   # splits per measure

# SPLITS_PER_BEAT = 1 / ((FAKE_BPM / 60 ) / CHART_FRAME_RATE)
# force 10 ms splits; Ex) 140 bpm /60-> 2.3333 bps /100-> .02333 'beats' per split 
FAKE_BPM = 60 * (CHART_FRAME_RATE / SPLITS_PER_BEAT)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='path to directory containing model files')
    parser.add_argument('--audio_file', type=str, help='path to audio file to use')
    parser.add_argument('--chart_format', type=str, default='ucs', choices=['ucs', 'ssc', 'both'], help='output format to use')
    parser.add_argument('--out_dir', type=str, required=True, help='where to save output files')
    parser.add_argument('--level', type=int, default=14, help='level of the chart to generate')
    parser.add_argument('--song_name', type=str, help='song name')
    parser.add_argument('--song_artist', type=str, help='song artist')
    parser.add_argument('--sampling', type=str, default='top-p', choices=['top-p', 'top-k', 'greedy', 'multinom'], 
                        help='choose the sampling strategy to use when generating the step sequence')
    parser.add_argument('-k', type=int, default=50, help='the k to use in top-k sampling')
    parser.add_argument('-p', type=int, default=0.9, help='the p to use in top-p sampling')

    return parser.parse_args()

def save_chart(chart_data, chart_type, chart_level, chart_format, song_name, artist, 
               audio_file, model_name, out_dir):
    if not os.path.isdir(out_dir):
        print(f'Creating output directory {out_dir}')
        os.makedirs(out_dir)

    audio_filename = audio_file.split(os.sep)[-1].split('.')[0]

    # convert each time to a beat/split
    # adapted from https://github.com/chrisdonahue/ddc/blob/master/infer/ddc_server.py (~221)
    ten_ms_frames_to_steps = {int(round(time * CHART_FRAME_RATE)) : step for time, step in chart_data}
    max_num_splits = max(ten_ms_frames_to_steps.keys())
    if max_num_splits % SPLIT_SUBDIV != 0:
        max_num_splits += SPLIT_SUBDIV - (max_num_splits % SPLIT_SUBDIV)
        
    if chart_type == 'pump-single':
        blank_note = '.' * 5
    elif chart_type == 'pump-double':
        blank_note = '.' * 10

    splits = []
    curr_holds = set()
    curr_steps = set()
    for i in range(max_num_splits):
        curr_note = ten_ms_frames_to_steps.get(i, blank_note)
        curr_note, _, _ = filter_steps(curr_note, curr_holds, curr_steps, dense=True)
        
        splits.append(curr_note)

    charts_to_save = []

    if chart_format == 'ucs' or chart_format == 'both':
        # 100ms for chart delay
        chart_attrs = { 'Format': 1, 'Mode': 'Single' if chart_type == 'pump-single' else 'Double',
                        'BPM': FAKE_BPM, 'Delay': 100, 'Beat': BEATS_PER_MEASURE, 'Split': SPLITS_PER_BEAT }

        chart_txt = ''
        for key, val in chart_attrs.items():
            chart_txt += f':{key}={val}\n'
        
        chart_txt += '\n'.join(splits)
        chart_fp = audio_filename + '.ucs'

        charts_to_save.append((chart_fp, chart_txt))

    # convert steps from ucs -> ssc format + save if needed
    if chart_format == 'ssc' or chart_format == 'both':
        chart_attrs = {'TITLE': song_name, 'ARTIST': song_name, 
            'MUSIC': os.path.join(out_dir, audio_filename), 'OFFSET': 0.0, 'BPMS': f'0.0={FAKE_BPM}', 
            'NOTEDATA': '', 'CHARTNAME': '', 'STEPSTYPE': chart_type, 'METER': str(level),
            'CREDIT': model_name, 'STOPS': '', 'DELAYS': ''
        }

        chart_txt = ''
        for key, val in chart_attrs.items():
            chart_txt += f'#{key}:{val};\n'

        chart_txt += '#NOTES:\n'
        for split in splits:
            chart_txt += ''.join([UCS_SSC_DICT[step] for step in split]) + '\n'
        chart_txt += ';\n'

        chart_fp = audio_filename + '.ssc'
        charts_to_save.append((chart_fp, chart_txt))

    for chart_fp, chart_txt in charts_to_save:
        with open(os.path.join(out_dir, chart_fp), 'w') as f:
            f.write(chart_txt)

    # copy the original audio file
    shutil.copy(audio_file, out_dir)

def generate_chart(placement_model, selection_model, audio_file, chart_type, 
                   chart_level, n_step_features, special_tokens, sampling, k, p, device):
    placement_model.eval()
    selection_model.eval()

    print('Generating chart - this may take a bit...')

    # store pairs of time (s) and step vocab indices
    chart_data = []

    # shape [3, ?, 80] -> [batch=1, 3, ?, 80]
    audio_feats, sample_rate = extract_audio_feats(audio_file)
    audio_feats = audio_feats.unsqueeze(0)
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
        print('Choosing placements...')
        states = placement_model.rnn.initStates(batch_size=1, device=device)
        audio_length = torch.tensor([n_audio_frames], dtype=torch.long, device=device)

        # [batch=1, n_audio_frames, 2] / [batch=1, n_audio_frames, hidden] / ...
        logits, placement_hiddens, _ = placement_model(audio_feats, chart_feats, states, audio_length)

        # placement predictions - [n_audio_frames] - 1 if a placement, 0 if empty
        placements = train_rnns.predict_placements(logits, [chart_level], audio_length).squeeze(0)

        placement_frames = (placements == 1).nonzero(as_tuple=False).flatten()
        num_placements = int(placements.sum().item())

        print(f'{num_placements} placements were chosen. Now selecting steps...')

        # Start generating the sequence of steps
        step_length = torch.ones(1, dtype=torch.long, device=device)
        hold_indices = set()
        step_indices = set()
        
        for i in trange(num_placements):
            placement_melframe = placement_frames[i].item()
            placement_time = util.convert_melframe_to_secs(placement_melframe, sample_rate)

            if i == 0:
                start_token = torch.zeros(1, 1, n_step_features, device=device)
                hidden, cell = selection_model.initStates(batch_size=1, device=device)
                logits, (hidden, cell) = selection_model(start_token, None, hidden, cell, step_length)
            else:
                next_token = next_token.unsqueeze(0).unsqueeze(0).float()
                placement_hidden = placement_hiddens[0, placement_melframe]
                logits, (hidden, cell) = selection_model(next_token, placement_hidden, hidden, cell, step_length)

            next_token_idx = predict_step(logits.squeeze(), sampling, k, p)
            
            # convert token index -> feature tensor -> str [ucs] representation
            next_token = step_index_to_features(next_token_idx, chart_type, special_tokens, device)
            next_token_str = step_features_to_str(next_token)
            (next_token_str,
             new_hold_indices,
             released_indices) = filter_steps(next_token_str, hold_indices, step_indices)

            # replace steps ('X') immediately before holds ('H') as ('M')
            #         holds ('H') immediately before steps ('X') as ('W')
            if new_hold_indices or released_indices:
                prev_step = chart_data[i - 1][1] 
                replacement = ''
                for k in range(len(prev_step)):
                    if k in new_hold_indices:
                        replacement += 'M'
                    elif k in released_indices:
                        replacement += 'W'
                    else:
                        replacement += prev_step[k]

                chart_data[i - 1][1] = replacement

            chart_data.append([placement_time, next_token_str])
   
    return chart_data

def filter_steps(token_str, hold_indices, step_indices, dense=False):
    # filter out step exceptions:
    #  - cannot release 'W' if not currently held
    #  - (retroactive) if 'H' appears directly after an 'X', change the 'X' -> 'M' 
    #    otherwise, change the first 'H' -> 'M'

    new_token_str = ''
    new_hold_indices = []
    released_indices = []

    for j in range(len(token_str)):
        token = token_str[j]
        curr_step_hold = j in hold_indices

        if token == 'W':
            if not curr_step_hold:
                token = '.'
            else:
                step_indices.discard(j)
                hold_indices.remove(j)
        elif token == 'H' and not curr_step_hold:
            hold_indices.add(j)
            if j not in step_indices:
                token = 'M'
            else:
                new_hold_indices.append(j)
                step_indices.remove(j)
        elif token == 'X':
            if curr_step_hold:
                released_indices.append(j)
                hold_indices.remove(j)

            step_indices.add(j)
        elif token == '.':
            step_indices.discard(j)

            if curr_step_hold:
                if dense:
                    token = 'H'
                else:
                    token = 'W'
                    hold_indices.remove(j)
             
        new_token_str += token

    return new_token_str, new_hold_indices, released_indices

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
        sorted_indices_to_remove = cumulative_probs > p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = 0

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

    print('Loading models....')
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

    if os.path.isfile(os.path.join(args.model_dir, 'special_tokens.json')):
        with open(os.path.join(args.model_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.loads(f.read())
    else:
        special_tokens = None

    # a list of pairs of (absolute time (s), step [ucs str format])
    chart_data = generate_chart(placement_model, selection_model, args.audio_file, model_summary['chart_type'],
                                args.level, model_summary['selection_input_size'], special_tokens, 
                                args.sampling, args.k, args.p, device)

    # convert indices -> chart output format + save to file
    save_chart(chart_data, model_summary['chart_type'], args.level, args.chart_format, args.song_name,
               args.song_artist, args.audio_file, model_summary['name'], args.out_dir)

if __name__ == '__main__':
    main()
