# Use a set of pretrained models to generate a step chart

import argparse
import json
import math
import os
import re
import shutil
from collections import OrderedDict

import torch
from tqdm import trange

import sys
sys.path.append(os.path.join('..', 'train'))
from hyper import (N_CHART_TYPES, N_LEVELS, CHART_FRAME_RATE, NUM_ARROW_STATES, SELECTION_INPUT_SIZES,
                   SUMMARY_SAVE, SPECIAL_TOKENS_SAVE, THRESHOLDS_SAVE)
from arrow_rnns import PlacementCLSTM, SelectionRNN
from arrow_transformer import ArrowTransformer
from extract_audio_feats import extract_audio_feats
from step_tokenize import get_state_indices, step_features_to_str, step_index_to_features, UCS_SSC_DICT
from predict_placements import predict_placements
import train_util

FILTERING_INDICES_PATH = 'filter_indices.json'

# default 4 beats per measure
BEATS_PER_MEASURE = 4
MIN_SPLITS_PER_BEAT = 4
HOLD_SPLIT_SKIP = 8

FAKE_SPLITS_PER_BEAT = 10 # ~ fake bpm = 600
FAKE_SPLIT_SUBDIV = FAKE_SPLITS_PER_BEAT * BEATS_PER_MEASURE
FAKE_BPM = 60 * (CHART_FRAME_RATE / FAKE_SPLITS_PER_BEAT)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='path to directory containing model files')
    parser.add_argument('--audio_file', type=str, help='path to audio file to use')
    parser.add_argument('--chart_format', type=str, default='ucs', choices=['ucs', 'ssc', 'both'], help='output format to use')
    parser.add_argument('--out_dir', type=str, required=True, help='where to save output files')
    parser.add_argument('--level', type=int, default=14, help='level of the chart to generate')
    parser.add_argument('--song_name', type=str, help='song name')
    parser.add_argument('--song_artist', type=str, help='song artist')
    parser.add_argument('--display_bpm', type=float, default=125, help='determine scroll speed/display')
    parser.add_argument('--sampling', type=str, default='top-k', choices=['top-p', 'top-k', 'beam-search', 'greedy', 'multinom'], 
                        help='choose the sampling strategy to use when generating the step sequence')
    parser.add_argument('-k', type=int, default=15, help='Sample steps from the top k candidates if using top-k')
    parser.add_argument('-p', type=float, default=0.075, help='Sample steps from the smallest set of candidates with cumulative prob. > p')
    parser.add_argument('-b', type=int, default=25, help='Beam size for beam search')

    return parser.parse_args()

def get_filter_indices(chart_type):
    # filtering indices for step selection (see filter_step_dist(..))
    # only compute once for base vocab sizes + save for effeciency

    # filter 1: The only possible next arrow for a current hold (started by 'M') is empty '.' or release 'W'
    # filter 2: A release note 'W' must follow a hold step 'M' (can't release without already holding)

    if os.path.isfile(FILTERING_INDICES_PATH):
        with open(FILTERING_INDICES_PATH, 'r') as f:
            step_filters = json.loads(f.read())
    else:
        step_filters = {'pump-single': {}, 'pump-double': {}}
        for mode in step_filters:
            curr_hold_filters, curr_empty_filters = {}, {}
            for j in range(SELECTION_INPUT_SIZES[mode] // NUM_ARROW_STATES):
                curr_hold_filters[str(j)] = get_state_indices(j, [1], mode)     # filter 1
                curr_empty_filters[str(j)] = get_state_indices(j, [3], mode)    # filter 2
            
            step_filters[mode]['hold'] = curr_hold_filters
            step_filters[mode]['empty'] = curr_empty_filters

        with open(FILTERING_INDICES_PATH, 'w') as f:
            f.write(json.dumps(step_filters))

    hold_filters = step_filters[chart_type]['hold']
    empty_filters = step_filters[chart_type]['empty']

    return hold_filters, empty_filters

def convert_splits_to_bpm(splits, bpm, blank_note):
    """
    Helper function to convert 10ms splits -> the given bpm as best as possible
        splits: list of notes representing all 10ms frames
    """
    #   bpm     1min       1s        10ms                                      bpm
    #  ----- * ------ * -------- * --------- -> beats per (10 ms) split =  ------------
    #   1min     60s     1000ms     1 split                                (60) * (100)
    # chart frame rate = 1/(^^^^^^^^^^^^^) = 100        
    # or alternatively, the number of 10ms splits ber beat = 6000 / bpm

    # Then we can divide every (6000/bpm) chunk of chart frames into beats, and
    # make subdivisions as necessary depending on the density

    # # of 10ms splits per beat; round to the nearest int -> may cause some time errors esp. for lower bpms
    # we want to fit this many frames into a single beat
    splits_per_beat = round((CHART_FRAME_RATE * 60) / bpm)  
    measure_length = splits_per_beat * BEATS_PER_MEASURE    # splits per measure

    if len(splits) % measure_length != 0:
        splits += [blank_note] * (measure_length - (len(splits) % measure_length))

    num_measures = len(splits) // measure_length

    # construct different chart sections (which may have different split divisions
    # depending on chart density - use the minimum number >= 4 to avoid long hold counts)
    # store pairs of [updated beatsplit, [notes for this section of splits]]
    chart_sections = []

    curr_beatsplit = 0
    hold_counter = 0

    for measure in range(num_measures):
        measure_start = measure * measure_length
        if curr_beatsplit > 0:
            chart_sections.append([curr_beatsplit, measure, []])
            
        for beat in range(BEATS_PER_MEASURE):
            beat_start = measure_start + (beat * splits_per_beat)
            curr_splits = splits[beat_start:beat_start + splits_per_beat]

            new_beatsplit = MIN_SPLITS_PER_BEAT
            split_interval = splits_per_beat // new_beatsplit

            # check if any splits have placements that don't line up with the
            # min splits per beat (4) - increase as necessary
            for k, steps in enumerate(curr_splits):
                if steps == blank_note:
                    continue
                elif re.search('[XMW]', steps):
                    hold_counter = 0
                elif re.search('H', steps):
                    # only keep 1 of every x splits (x * 10 ms) that only has hold notes
                    hold_counter += 1
                    if hold_counter == HOLD_SPLIT_SKIP:
                        hold_counter = 0
                    else:
                        continue

                # in the worst case, we need to keep all the original splits
                while k % split_interval != 0 and new_beatsplit < splits_per_beat:
                    new_beatsplit += 1
                    split_interval = math.ceil(splits_per_beat / new_beatsplit)

            # if different number of splits than before, start a new chart section
            if new_beatsplit  != curr_beatsplit:
                curr_beatsplit = new_beatsplit
                abs_measure = measure + (beat / BEATS_PER_MEASURE)
                chart_sections.append([curr_beatsplit, abs_measure, []])

            for k, steps in enumerate(curr_splits):
                if k % split_interval == 0:
                    chart_sections[-1][-1].append(steps)

                    # at this point the absolute time should not have changed by much, i.e.
                    old_time = (beat_start + k) / CHART_FRAME_RATE
                    new_time = ((measure * BEATS_PER_MEASURE + beat + ((k + 1) / splits_per_beat)) / bpm) * 60
                    if abs(round(old_time, 2) - round(new_time, 2)) > 0.04:
                        print(f'Warning, conversion difference > 40ms; {old_time} !~ {new_time}')

    return splits_per_beat, chart_sections 

def save_chart(chart_data, chart_type, chart_level, chart_format, display_bpm,
               song_name, artist, audio_file, model_name, out_dir):
    if not os.path.isdir(out_dir):
        print(f'Creating output directory {out_dir}')
        os.makedirs(out_dir)

    audio_ext = audio_file.split('.')[-1]
    audio_filename = song_name.lower()

    # convert each time to a beat/split
    # adapted from https://github.com/chrisdonahue/ddc/blob/master/infer/ddc_server.py (~221)
    ten_ms_frames_to_steps = {int(round(time * CHART_FRAME_RATE)) : step for time, step in chart_data}
            
    if chart_type == 'pump-single':
        blank_note = '.' * 5
    elif chart_type == 'pump-double':
        blank_note = '.' * 10

    splits = []
    curr_holds = set()
    for i in range(max(ten_ms_frames_to_steps.keys())):
        curr_note = ten_ms_frames_to_steps.get(i, blank_note)

        filtered_note = ''

        # fill in frames in between hold steps
        for j, step in enumerate(curr_note):
            if step == '.' and j in curr_holds:
                step = 'H'
            elif step == 'M':
                curr_holds.add(j)
            elif step == 'W':
                curr_holds.discard(j)
            
            filtered_note += step
            
        splits.append(filtered_note)

    charts_to_save = []

    if chart_format == 'ucs' or chart_format == 'both':
        # convert splits to the given bpm; return pairs of updated beatsplits/chart notes
        splits_per_beat, chart_sections = convert_splits_to_bpm(splits, display_bpm, blank_note)
        chart_attrs = { 'Format': 1, 'Mode': 'Single' if chart_type == 'pump-single' else 'Double'}

        chart_txt = ''
        for key, val in chart_attrs.items():
            chart_txt += f':{key}={val}\n'
        
        for i, (beatsplit, _, notes) in enumerate(chart_sections):
            if notes:
                chart_txt += f':BPM={display_bpm}\n:Delay=0\n:Beat={BEATS_PER_MEASURE}\n:Split={beatsplit}\n'        
                chart_txt += '\n'.join(notes) + '\n'

        chart_fp = audio_filename + '.ucs'
        charts_to_save.append((chart_fp, chart_txt))

    # convert steps from ucs -> ssc format if needed
    # custom bpms for ssc format not supported due to conversion issues (could be potentially fixed ?)
    if chart_format == 'ssc' or chart_format == 'both':
        notes_txt = '\n'
        for j, split in enumerate(splits):
            notes_txt += ''.join([UCS_SSC_DICT[step] for step in split]) + '\n'

            # separate measures by commas
            if (j + 1) % FAKE_SPLIT_SUBDIV == 0:
                notes_txt += ',\n'

        chart_attrs = {'TITLE': song_name, 'ARTIST': artist, 'OFFSET': 0.0, 'BPMS': f'0.0={FAKE_BPM}',
                       'MUSIC': audio_filename + '.' + audio_ext, 'NOTEDATA': '', 'CHARTNAME': '',
                       'STEPSTYPE': chart_type, 'METER': str(chart_level), 'CREDIT': model_name,
                       'STOPS': '', 'DELAYS': '', 'NOTES': notes_txt}

        chart_txt = ''
        for key, val in chart_attrs.items():
            chart_txt += f'#{key}:{val};\n'

        chart_fp = audio_filename + '.ssc'
        charts_to_save.append((chart_fp, chart_txt))

    for chart_fp, chart_txt in charts_to_save:
        with open(os.path.join(out_dir, chart_fp), 'w') as f:
            f.write(chart_txt)

    # copy the original audio file
    shutil.copyfile(audio_file, os.path.join(out_dir, audio_filename + '.'  + audio_ext))

    print(f'Saved to {out_dir}')

def generate_chart(placement_model, selection_model, audio_file, chart_type, chart_level, vocab_size, thresholds,
                   n_step_features, special_tokens, conditioned, sampling, k, p, b, device=torch.device('cpu')):
    print('Generating chart - this may take a bit...')

    (placements,
     peaks,
     placement_hiddens,
     sample_rate) = generate_placements(placement_model, audio_file, chart_type,
                                        chart_level, thresholds, n_step_features, device)

    if not conditioned:
        placement_hiddens = None

    chart_data = generate_steps(selection_model, placements, placement_hiddens, vocab_size, n_step_features,
                                chart_type, sample_rate, special_tokens, sampling, k, p, b, device)

    return chart_data, peaks
        
def generate_placements(placement_model, audio_file, chart_type, chart_level, thresholds, 
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
        placements, probs = predict_placements(logits, [chart_level], audio_length, get_probs=True, thresholds=thresholds)

        placements = placements.squeeze(0)
        probs = probs.squeeze(0).tolist()
        
        placement_hiddens = placement_hiddens[0, (placements == 1).nonzero(as_tuple=False).flatten()]


    peaks = [(i / CHART_FRAME_RATE, prob) for i, prob in enumerate(probs)]

    return placements, peaks, placement_hiddens, sample_rate    

def generate_steps(selection_model, placements, placement_hiddens, vocab_size, n_step_features, chart_type,
                   sample_rate, special_tokens, sampling, k, p, b, device=torch.device('cpu')):
    placement_frames = (placements == 1).nonzero(as_tuple=False).flatten()
    num_placements = int(placements.sum().item())

    print(f'{num_placements} placements were chosen. Now selecting steps...')

    # store pairs of time (s) and step vocab indices
    chart_data = []

    num_arrows = n_step_features // NUM_ARROW_STATES

    hold_filters, empty_filters = get_filter_indices(chart_type)

    # Start generating the sequence of steps
    step_length = torch.ones(1, dtype=torch.long, device=device)
    hold_indices = [set() for _ in range(b)] if sampling == 'beam-search' else set()

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
            placement_hidden = placement_hiddens[i] if conditioned and i > 0 else None
            
            if sampling == 'beam-search':
                if i == 0:
                    beams.append([[0], 0.0, hidden.clone(), cell.clone()])

                candidates, curr_states = get_beam_candidates(selection_model, beams, b, placement_hidden, step_length, 
                                                              num_arrows, hold_indices, chart_type, special_tokens, 
                                                              hold_filters, empty_filters, device)
                
                # if the last element in the sequence, keep the one with the best score
                if i == num_placements - 1:
                   save_best_beam(candidates[0], placement_times, chart_data, chart_type, special_tokens, device)
                else:
                    for seq, score, beam_idx in candidates:
                        curr_hidden, curr_cell = curr_states[beam_idx]
                        beams[beam_idx] = [seq, score, curr_hidden, curr_cell]
            else:
                curr_token = next_token_feats.unsqueeze(0).unsqueeze(0).float() if i > 0 else start_token
                logits, (hidden, cell) = selection_model(curr_token, placement_hidden, hidden, cell, step_length)
                next_token_idx = predict_step(logits.squeeze(), sampling, k, p, hold_indices, 
                                              num_arrows, hold_filters, empty_filters)
            
                # convert token index -> feature tensor -> str [ucs] representation
                next_token_feats = step_index_to_features(next_token_idx, chart_type, special_tokens, device)
                next_token_str = step_features_to_str(next_token_feats)
                next_token_str, new_hold_indices = filter_steps(next_token_str, hold_indices)
                
                chart_data.append([placement_time, next_token_str])

    return chart_data

def get_beam_candidates(selection_model, beams, beam_width, placement_hidden, step_length, 
                        num_arrows, hold_indices, chart_type, special_tokens, hold_filters, 
                        empty_filters, device):
    # store expanded seqs, scores, + original beam idx
    candidates, curr_states = [], []
    for z, (seq, beam_score, hidden, cell) in enumerate(beams):
        curr_token = step_index_to_features(seq[-1], chart_type, special_tokens, device)
        curr_token = curr_token.unsqueeze(0).unsqueeze(0).float()

        logits, (hidden, cell) = selection_model(curr_token, placement_hidden, hidden, cell, step_length)
        curr_states.append((hidden, cell))    

        curr_candidates = expand_beam(logits.squeeze(), seq, beam_score, z, hold_indices,
                                      num_arrows, hold_filters, empty_filters)

        candidates.extend(curr_candidates)

    # sort by beam score (accumulated (abs) log probs) -> keep lowest b candidates
    candidates = sorted(candidates, key=lambda x:x[1])[:beam_width]

    return candidates, curr_states

def expand_beam(logits, sequence, beam_score, beam_idx, hold_indices, num_arrows, hold_filters, empty_filters):
    # logits: [vocab_size]
    dist = torch.nn.functional.softmax(logits, dim=-1)
    dist = filter_step_dist(dist, hold_indices[beam_idx], num_arrows, hold_filters, empty_filters)
    expanded = []
    for j in range(logits.size(0)):
        new_seq = sequence + [j]

        curr_prob = dist[j] if dist[j] > 0 else 1e-16
        new_score = beam_score - math.log(curr_prob)
        expanded.append([new_seq, new_score, beam_idx])

    return expanded

def save_best_beam(best, placement_times, chart_data, chart_type, special_tokens, device):
    seq, _, _ = best
    hold_indices = set()
    for m in range(1, len(seq)):
        token_feats = step_index_to_features(seq[m], chart_type, special_tokens, device)
        token_str = step_features_to_str(token_feats)
        token_str, new_holds = filter_steps(token_str, hold_indices)
        
        chart_data.append([placement_times[m - 1], token_str])

def filter_steps(token_str, hold_indices):
    # filter out step exceptions + track holds
    new_token_str = ''

    for j in range(len(token_str)):
        token = token_str[j]

        #  - cannot release 'W' if not currently held
        if token == 'W':
            if j not in hold_indices:
                token = '.'
            else:
                hold_indices.remove(j)
        elif token == 'M':
            if j in hold_indices:
                token = '.'
        elif token == 'X':
            if j in hold_indices:
                token = '.'
             
        new_token_str += token

    return new_token_str

def filter_step_dist(dist, hold_indices, num_arrows, hold_filters, empty_filters):
    """Filter step distributions to exclude impossible step sequences; see get_filter_indices()"""
    # filter 1: The only possible next arrow states for a current hold is another hold 'H' or release 'W'
    # filter 2: A hold note 'H' must follow a step 'X' or another hold 'H'
    # filter 3: A release note 'W' must follow a step 'X' (treat as the hold's start) or another hold 'H'
    for j in range(num_arrows):
        if j in hold_indices:
            # filter step combinations where the jth step is step 'X' if held
            dist[hold_filters[str(j)]] = 0.0
        else:
            # filter steps where jth step is a release 'W' if not held
            dist[empty_filters[str(j)]] = 0.0

    return dist

def predict_step(logits, sampling, k, p, hold_indices, num_arrows, hold_filters, empty_filters):
    """predict the next step given model logits and a sampling strategy"""
    # shape: [vocab_size]
    dist = torch.nn.functional.softmax(logits, dim=-1)
    dist = filter_step_dist(dist, hold_indices, num_arrows, hold_filters, empty_filters)
    
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
        logits[indices_to_remove] = float('-inf')

        filtered_dist = torch.nn.functional.softmax(logits, dim=-1)
        filtered_dist = filter_step_dist(filtered_dist, hold_indices, num_arrows, hold_filters, empty_filters)
        pred_idx = torch.multinomial(filtered_dist, num_samples=1)              
    elif sampling == 'greedy':
        # take the most likely token
        pred_idx = torch.topk(dist, k=1, dim=-1)[1][0]
    elif sampling == 'multinom':
        # sample from the bare distribution
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

    if os.path.isfile(os.path.join(model_dir, SPECIAL_TOKENS_SAVE)):
        with open(os.path.join(model_dir, SPECIAL_TOKENS_SAVE), 'r') as f:
            special_tokens = json.loads(f.read())
    else:
        special_tokens = None

    with open(os.path.join(model_dir, THRESHOLDS_SAVE), 'r') as f:
        placement_thresholds = json.loads(f.read())
   
    return placement_model, selection_model, special_tokens, placement_thresholds

def main():
    args = parse_args()

    device = torch.device('cpu')

    print('Loading models....')
    with open(os.path.join(args.model_dir, SUMMARY_SAVE), 'r') as f:
        model_summary = json.loads(f.read())

    placement_model, selection_model, special_tokens, thresholds = get_gen_config(model_summary, args.model_dir, device)
    conditioned = model_summary['conditioning']

    print(f'Generating a {model_summary["chart_type"].split("-")[-1]} {args.level} chart for {args.song_name}')
    print(f'Decoding strategy: {args.sampling} ; k: {args.k}, p: {args.p}, b: {args.b}')

    vocab_size = model_summary['vocab_size']

    # a list of pairs of (absolute time (s), step [ucs str format])
    chart_data, _ = generate_chart(placement_model, selection_model, args.audio_file, model_summary['chart_type'],
                                   args.level, vocab_size, thresholds, model_summary['selection_input_size'],
                                   special_tokens, conditioned, args.sampling, args.k, args.p, args.b, device)

    if chart_data:
        # convert indices -> chart output format + save to file
        save_chart(chart_data, model_summary['chart_type'], args.level, args.chart_format,
                args.display_bpm, args.song_name, args.song_artist, args.audio_file,
                model_summary['name'], args.out_dir)

if __name__ == '__main__':
    main()
