# Use a set of pretrained models to generate a step chart

import argparse
import json
import math
import os
import shutil

import torch
from tqdm import trange

import sys
sys.path.append(os.path.join('..', 'train'))
from hyper import N_CHART_TYPES, N_LEVELS, CHART_FRAME_RATE, NUM_ARROW_STATES, SELECTION_INPUT_SIZES
from arrow_rnns import PlacementCLSTM, SelectionRNN
from arrow_transformer import ArrowTransformer
from extract_audio_feats import extract_audio_feats
from step_tokenize import get_state_indices, step_features_to_str, step_index_to_features, UCS_SSC_DICT
from predict_placements import predict_placements
import train_util

BEATS_PER_MEASURE = 1
SPLITS_PER_BEAT = 10
SPLIT_SUBDIV = SPLITS_PER_BEAT * BEATS_PER_MEASURE   # splits per measure

# SPLITS_PER_BEAT = 1 / ((FAKE_BPM / 60 ) / CHART_FRAME_RATE)
# force 10 ms splits; Ex) 140 bpm /60-> 2.3333 bps /100-> .02333 'beats' per split 
# 10 spb > ~600bpm - only affects default scroll speed
# TODO if bpm provided, convert/approximate subdivisions
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
    parser.add_argument('--sampling', type=str, default='top-p', choices=['top-p', 'top-k', 'beam-search', 'greedy', 'multinom'], 
                        help='choose the sampling strategy to use when generating the step sequence')
    parser.add_argument('-k', type=int, default=10, help='Sample steps from the top k candidates if using top-k')
    parser.add_argument('-p', type=float, default=0.1, help='Sample steps from the smallest set of candidates with cumulative prob. > p')
    parser.add_argument('-b', type=int, default=10, help='Beam size for beam search')

    return parser.parse_args()

def save_chart(chart_data, chart_type, chart_level, chart_format, song_name, artist, 
               audio_file, model_name, out_dir):
    if not os.path.isdir(out_dir):
        print(f'Creating output directory {out_dir}')
        os.makedirs(out_dir)

    audio_ext = audio_file.split('.')[-1]
    audio_filename = song_name.lower()

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
    for i in range(max_num_splits):
        curr_note = ten_ms_frames_to_steps.get(i, blank_note)

        filtered_note = ''

        # fill frames in between hold steps
        for j, step in enumerate(curr_note):
            if step == '.' and j in curr_holds:
                step = 'H'
            elif step == 'M':
                curr_holds.add(j)
            elif step == 'W':
                curr_holds.remove(j)
            
            filtered_note += step
            
        splits.append(filtered_note)

    charts_to_save = []

    if chart_format == 'ucs' or chart_format == 'both':
        # 100ms for delay
        chart_attrs = { 'Format': 1, 'Mode': 'Single' if chart_type == 'pump-single' else 'Double',
                        'BPM': FAKE_BPM, 'Delay': 0, 'Beat': BEATS_PER_MEASURE, 'Split': SPLITS_PER_BEAT }

        chart_txt = ''
        for key, val in chart_attrs.items():
            chart_txt += f':{key}={val}\n'
        
        chart_txt += '\n'.join(splits)
        chart_fp = audio_filename + '.ucs'

        charts_to_save.append((chart_fp, chart_txt))

    # convert steps from ucs -> ssc format + save if needed ######## TODO fix output ########
    if chart_format == 'ssc' or chart_format == 'both':
        chart_attrs = {'TITLE': song_name, 'ARTIST': song_name, 
            'MUSIC': os.path.join(out_dir, audio_filename), 'OFFSET': 0.0, 'BPMS': f'0.0={FAKE_BPM}', 
            'NOTEDATA': '', 'CHARTNAME': '', 'STEPSTYPE': chart_type, 'METER': str(chart_level),
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
    shutil.copyfile(audio_file, os.path.join(out_dir, audio_filename + '.'  + audio_ext))

    print(f'Saved to {out_dir}')

def generate_chart(placement_model, selection_model, audio_file, chart_type, chart_level, 
                   n_step_features, special_tokens, conditioned, sampling, k, p, b, device=torch.device('cpu')):
    print('Generating chart - this may take a bit...')

    (placements,
     peaks,
     placement_hiddens,
     sample_rate) = generate_placements(placement_model, audio_file, chart_type,
                                        chart_level, n_step_features, device)

    if not conditioned:
        placement_hiddens = None

    chart_data = generate_steps(selection_model, placements, placement_hiddens, n_step_features,
                                chart_type, sample_rate, special_tokens, sampling, k, p, b, device)

    return chart_data, peaks
        
def generate_placements(placement_model, audio_file, chart_type, chart_level, 
                        n_step_features, device=torch.device('cpu')):
    placement_model.eval()

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

        # placement predictions - [batch=1, n_audio_frames] - 1 if a placement, 0 if empty
        placements, probs = predict_placements(logits, [chart_level], audio_length, get_probs=True)

        placements = placements.squeeze(0)
        probs = probs.squeeze(0).tolist()

    peaks = [(i / CHART_FRAME_RATE, prob) for i, prob in enumerate(probs)]

    return placements, peaks, placement_hiddens, sample_rate    

def generate_steps(selection_model, placements, placement_hiddens, n_step_features, chart_type,
                   sample_rate, special_tokens, sampling, k, p, b, device=torch.device('cpu')):
    placement_frames = (placements == 1).nonzero(as_tuple=False).flatten()
    num_placements = int(placements.sum().item())

    print(f'{num_placements} placements were chosen. Now selecting steps...')

    # store pairs of time (s) and step vocab indices
    chart_data = []

    # Start generating the sequence of steps
    step_length = torch.ones(1, dtype=torch.long, device=device)
    hold_indices = [set() for _ in range(b)] if sampling == 'beam-search' else set()
    step_indices = [set() for _ in range(b)] if sampling == 'beam-search' else set()

    conditioned = placement_hiddens is not None

    selection_model.eval()

    # for beam search, track the b most likely token sequences + 
    # their (log) probabilities at each step
    beams = []
    placement_times = []

    with torch.no_grad():
        start_token = torch.zeros(1, 1, n_step_features, device=device)
        hidden, cell = selection_model.initStates(batch_size=1, device=device)
    
        for i in trange(num_placements):
            placement_melframe = placement_frames[i].item()
            placement_time = train_util.convert_melframe_to_secs(placement_melframe, sample_rate)
            placement_times.append(placement_time)
            placement_hidden = placement_hiddens[0, placement_melframe] if conditioned and i > 0 else None

            if sampling == 'beam-search':
                if i == 0:
                    beams.append([[0], 0.0, hidden.clone(), cell.clone()])

                # store expanded seqs, scores, + original beam idx
                candidates, curr_states = [], []
                for z, (seq, beam_score, hidden, cell) in enumerate(beams):
                    curr_token = step_index_to_features(seq[-1], chart_type, special_tokens, device)
                    curr_token = curr_token.unsqueeze(0).unsqueeze(0).float()

                    logits, (hidden, cell) = selection_model(curr_token, placement_hidden, hidden, cell, step_length)
                    curr_states.append((hidden, cell))    

                    candidates.extend(expand_beam(logits.squeeze(), seq, beam_score, z, hold_indices, step_indices,
                        chart_type, special_tokens))

                # sort by beam score (accumulated (abs) log probs) -> keep lowest b candidates
                candidates = sorted(candidates, key=lambda x:x[1])[:b]
                
                # if the last element in the sequence, keep the one with the best score
                if i == num_placements - 1:
                    seq, _, _ = candidates[0]
                    hold_indices = set()
                    step_indices = set()
                    for m in range(1, len(seq)):
                        token_feats = step_index_to_features(seq[m], chart_type, special_tokens, device)
                        token_str = step_features_to_str(token_feats)
                        token_str, new_holds = filter_steps(token_str, hold_indices, step_indices)

                        if m > 1:
                            replace_steps(new_holds, chart_data, m - 1)
                        
                        chart_data.append([placement_times[m - 1], token_str])
                else:
                    for seq, score, beam_idx in candidates:
                        curr_hidden, curr_cell = curr_states[beam_idx]
                        beams[beam_idx] = [seq, score, curr_hidden, curr_cell]
            else:
                curr_token = next_token_feats.unsqueeze(0).unsqueeze(0).float() if i > 0 else start_token
                logits, (hidden, cell) = selection_model(curr_token, placement_hidden, hidden, cell, step_length)
                next_token_idx = predict_step(logits.squeeze(), sampling, k, p, hold_indices, step_indices, 
                                              chart_type, special_tokens)
            
                # convert token index -> feature tensor -> str [ucs] representation
                next_token_feats = step_index_to_features(next_token_idx, chart_type, special_tokens, device)
                next_token_str = step_features_to_str(next_token_feats)
                next_token_str, new_hold_indices = filter_steps(next_token_str, hold_indices, step_indices)

                replace_steps(new_hold_indices, chart_data, i)
                
                chart_data.append([placement_time, next_token_str])

    return chart_data

def replace_steps(new_hold_indices, chart_data, i):
    # replace steps ('X') immediately before holds ('H') as ('M')
    if new_hold_indices:
        prev_step = chart_data[i - 1][1] 
        replacement = ''
        for k in range(len(prev_step)):
            if k in new_hold_indices:
                replacement += 'M'
            else:
                replacement += prev_step[k]

        chart_data[i - 1][1] = replacement

def filter_steps(token_str, hold_indices, step_indices):
    # filter out step exceptions
    new_token_str = ''
    new_hold_indices = []
    released_indices = []

    for j in range(len(token_str)):
        token = token_str[j]

        #  - cannot release 'W' if not currently held
        if token == 'W':
            if j in step_indices:
                # if previous step 'X', change X -> M, and leave this as 'W' (fill in 'H' later)
                new_hold_indices.append(j)
                step_indices.remove(j)
            else:
                # should have  j in hold_indices
                hold_indices.remove(j)
        elif token == 'H' and j not in hold_indices:
            hold_indices.add(j)
            #  - (retroactive) if 'H' appears directly after an 'X', change the 'X' -> 'M' (hold start)
            if j in step_indices:
                new_hold_indices.append(j)
                step_indices.remove(j)
        elif token == 'X':
            step_indices.add(j)
        elif token == '.':
            step_indices.discard(j)
             
        new_token_str += token

    return new_token_str, new_hold_indices

def expand_beam(logits, sequence, beam_score, beam_idx, hold_indices, step_indices, chart_type, special_tokens):
    # logits: [vocab_size]
    dist = torch.nn.functional.softmax(logits, dim=-1)
    dist = filter_step_dist(dist, hold_indices[beam_idx], step_indices[beam_idx], chart_type, special_tokens)
    expanded = []
    for j in range(logits.size(0)):
        new_seq = sequence + [j]

        curr_prob = dist[j] if dist[j] > 0 else 1e-16
        new_score = beam_score - math.log(curr_prob)
        expanded.append([new_seq, new_score, beam_idx])

    return expanded

def filter_step_dist(dist, hold_indices, step_indices, chart_type, special_tokens):
    """Filter step distributions to exclude impossible step sequences"""
    # filter 1: The only possible next arrow states for a current hold is another hold 'H' or release 'W'
    # filter 2: A hold note 'H' must follow a step 'X' or another hold 'H'
    # filter 3: A release note 'W' must follow a step 'X' (treat as the hold's start) or another hold 'H'
    for j in range(SELECTION_INPUT_SIZES[chart_type] // NUM_ARROW_STATES):
        # filter step combinations where the jth step is empty or a step (states = 0, 1)
        if j in hold_indices:
            filtered_indices = get_state_indices(j, [0, 1], chart_type, special_tokens)
            dist[filtered_indices] = 0.0

        # if arrow j neither preceded by 'H' or 'X', filter out holds (state = 2) and releases (state = 3)
        elif j not in step_indices:
            filtered_indices = get_state_indices(j, [2, 3], chart_type, special_tokens)
            dist[filtered_indices] = 0.0

    return dist

def predict_step(logits, sampling, k, p, hold_indices, step_indices, chart_type, special_tokens):
    """predict the next step given model logits and a sampling strategy"""
    # shape: [vocab_size]
    dist = torch.nn.functional.softmax(logits, dim=-1)
    
    if sampling == 'top-k':
        # sample from top k probabilites + corresponding indices (sorted)    
        dist = filter_step_dist(dist, hold_indices, step_indices, chart_type, special_tokens)
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
        logits[indices_to_remove] = float('-inf')

        filtered_dist = filter_step_dist(torch.nn.functional.softmax(logits, dim=-1), hold_indices,
                                         step_indices, chart_type, special_tokens)
        pred_idx = torch.multinomial(filtered_dist, num_samples=1)              
    elif sampling == 'greedy':
        # take the most likely token
        dist = filter_step_dist(dist, hold_indices, step_indices, chart_type, special_tokens)
        pred_idx = torch.topk(dist, k=1, dim=-1)[1][0]
    elif sampling == 'multinom':
        # sample from the bare distribution
        dist = filter_step_dist(dist, hold_indices, step_indices, chart_type, special_tokens)
        pred_idx = torch.multinomial(dist, num_samples=1)

    return pred_idx
    
def get_gen_config(model_summary, model_dir, device=torch.device('cpu')):
    if model_summary['type'] == 'rnns':
        placement_model = PlacementCLSTM(model_summary['placement_channels'], model_summary['placement_filters'], 
                                         model_summary['placement_kernels'], model_summary['placement_pool_kernel'],
                                         model_summary['placement_pool_stride'], model_summary['placement_lstm_layers'],
                                         model_summary['placement_input_size'], model_summary['hidden_size']).to(device)

        selection_model = SelectionRNN(model_summary['selection_lstm_layers'], model_summary['selection_input_size'], 
                                       model_summary['vocab_size'], model_summary['hidden_size'],
                                       model_summary['selection_hidden_wt']).to(device)

        # loads the state dicts from the .bin files in model_dir/
        train_util.load_save(model_dir, False, placement_model, selection_model, device)
    elif model_summary['type'] == 'transformer':
        pass

    if os.path.isfile(os.path.join(model_dir, 'special_tokens.json')):
        with open(os.path.join(model_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.loads(f.read())
    else:
        special_tokens = None
        
    return placement_model, selection_model, special_tokens

def main():
    args = parse_args()

    device = torch.device('cpu')

    print('Loading models....')
    with open(os.path.join(args.model_dir, 'summary.json'), 'r') as f:
        model_summary = json.loads(f.read())

    placement_model, selection_model, special_tokens = get_gen_config(model_summary, args.model_dir, device)
    conditioned = model_summary['conditioning']

    print(f'Generating a {model_summary["chart_type"].split("-")[-1]} {args.level} chart for {args.song_name}')
    print(f'Decoding strategy: {args.sampling} ; k: {args.k}, p: {args.p}, b: {args.b}')

    # a list of pairs of (absolute time (s), step [ucs str format])
    chart_data, _ = generate_chart(placement_model, selection_model, args.audio_file, model_summary['chart_type'],
                                   args.level, model_summary['selection_input_size'], special_tokens, conditioned,
                                   args.sampling, args.k, args.p, args.b, device)

    # convert indices -> chart output format + save to file
    save_chart(chart_data, model_summary['chart_type'], args.level, args.chart_format, args.song_name,
               args.song_artist, args.audio_file, model_summary['name'], args.out_dir)

if __name__ == '__main__':
    main()
