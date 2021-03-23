# step <-> index tokenization functions

import itertools
import math

import torch

from hyper import (NUM_ARROW_STATES, SELECTION_INPUT_SIZES, SELECTION_VOCAB_SIZES, 
                   MAX_ACTIVE_ARROWS, NUM_ACTIVE_STATES, TIME_FEATURES)

UCS_SSC_DICT = {
    '.': '0',   # no step
     'X': '1',   # normal step
     'M': '2',   # start hold
     'H': '0',   # hold (0 between '2' ... '3' in ssc)
     'W': '3',   # release hold
}

# symbols used in step representation
UCS_STATE_DICT = {
    '.': 0,		# off
    'X': 1,		# on
    'M': 2,		# hold start
    'H': 3,		# held
    'W': 4,		# hold release
}

STATE_TO_UCS = {
    0: '.',
    1: 'X',
    2: 'M',
    3: 'H',
    4: 'W'
}

# convert a sequence of steps ['00100', '10120', ...] -> input tensor
def sequence_to_tensor(sequence):
    # shape [abs # of frames, num arrow states * # arrows]
    #   (for each arrow, mark 1 of n possible states - off, step, hold start, hold, release)
    #	(should be already converted to (reduced) UCS notation)
    # eg. ['X000H', '0X00W'] -> [[0, 1, 0, 0, 0, 0, 0, 0, ..., 0, 0, 1, 0] 
    #                             -downleft-   -upleft- ....   -downright-
    #                            [0, 0, 0, 0, 0, 1, 0, 0, ..., 0, 0, 0, 1]]
    step_tensors = []

    num_steps = len(sequence[0])

    for step in sequence:
        symbol_tensors = []
        for i, symbol in enumerate(step):
            arrow_i_state = torch.zeros(NUM_ARROW_STATES)
            arrow_i_state[UCS_STATE_DICT[symbol]] = 1
            symbol_tensors.append(arrow_i_state)

        # convert symbols -> concatenated one hot encodings
        step_tensors.append(torch.cat(symbol_tensors))

    return torch.cat(step_tensors).view(-1, NUM_ARROW_STATES * num_steps)

def step_sequence_to_targets(step_input, chart_type, special_tokens):
    """
    given a (sequence) of step inputs, return a tensor containing the corresponding vocabulary indices
        in: step_input - shape [seq length, chart_features], tensor representations of (nonempty) steps
            chart_type - 'pump-single' or 'pump-double'
            special_tokens - predefined special token indices/values in the chart's dataset,
                             use for doubles charts with 5+ activated arrows at a time/singles for 4+
        out: targets - shape [seq length], values in range [0, vocab size - 1]
    """
    num_arrows = step_input.size(1) // NUM_ARROW_STATES
    targets = torch.zeros(step_input.size(0), dtype=torch.long)
    n_special_tokens = 0

    for s in range(step_input.size(0)):
        # index in order of: num active arrows [0, ..., num_arrows - 1]
        # 				  -> arrow indices [0, ..., num_arrows - 1] x num_active_arrows
        #				  -> arrow states [1, ..., num_arrow_states - 1] (ignore off)

        # step_index =  SUM_{k=0->num_active - 1} (num_arrows choose k) * 3 ^ k +
        #				SUM_{i' < i} ... SUM_{j' < j} NUM_ACTIVE_STATES ^ num_active +
        #				SUM_{x1, .., x_num_active} (x_i - 1) * NUM_ACTIVE_STATES ^ i (base NUM_ACTIVE_STATES R -> L)
        #	[i' = first index L->R, i = true first index, j', j = zth index, z = num active arrows]
        idx = 0

        active_arrow_states = []
        active_arrow_indices = []
        for i in range(num_arrows):
            start = i * NUM_ARROW_STATES
            arrow_state = (step_input[s, start:start + NUM_ARROW_STATES] == 1).nonzero(as_tuple=False).flatten()

            if arrow_state != 0:
                active_arrow_indices.append(i)
                active_arrow_states.append(arrow_state)

        num_active_arrows = len(active_arrow_indices)

        # if num active exceed maximum, add to end of special tokens
        if num_active_arrows > MAX_ACTIVE_ARROWS[chart_type]:
            curr_step = step_features_to_str(step_input[s])
            if curr_step not in special_tokens.values():
                special_idx = SELECTION_VOCAB_SIZES[chart_type] + len(special_tokens)
                special_tokens[special_idx] = curr_step
                n_special_tokens += 1
            else:
                for k,v in special_tokens.items():
                    if v == curr_step:
                        special_idx = k
                        break
            
            targets[s] = int(special_idx)
            continue

        for num_active in range(num_active_arrows):
            idx += math.comb(num_arrows, num_active) * (NUM_ACTIVE_STATES ** num_active)

        first_possible_idx = 0
        for k, arrow_idx in enumerate(active_arrow_indices):
            if arrow_idx > first_possible_idx:
                # count over the possible arrangements of steps that use the skipped indices
                for skipped_idx in range(first_possible_idx, arrow_idx):
                    idx += (NUM_ACTIVE_STATES ** num_active_arrows) * calc_arrangements(skipped_idx, len(active_arrow_indices) - k, num_arrows - 1)
            
            first_possible_idx = arrow_idx + 1
        
        # base 3 R -> L
        active_arrow_states.reverse()
        for a, state in enumerate(active_arrow_states):
            idx += (state - 1) * (NUM_ACTIVE_STATES ** a)

        targets[s] = idx

    return targets, n_special_tokens


def step_features_to_str(features, out_format='ucs'):
    """ convert step features to their string representation"""
    num_arrows = features.size(0) // NUM_ARROW_STATES
    result = ''

    if out_format == 'ucs':
        for arrow in range(num_arrows):
            start = arrow * NUM_ARROW_STATES
            state_idx = (features[start:start + NUM_ARROW_STATES] == 1).nonzero(as_tuple=False).flatten()

            result += STATE_TO_UCS[state_idx.item()]
    elif out_format == 'ssc':
        raise NotImplementedError

    return result

def step_index_to_features(index, chart_type, special_tokens, device):
    """ convert a step index to its corresponding feature tensor """

    if special_tokens and index in special_tokens:
        return sequence_to_tensor([special_tokens[index]])

    # perform 'inverse' of step_sequence_to_targets()
    features = torch.zeros(SELECTION_INPUT_SIZES[chart_type] - TIME_FEATURES, dtype=torch.long, device=device)
    num_arrows = features.size(0) // NUM_ARROW_STATES
    off_indices = torch.tensor([arrow * NUM_ARROW_STATES for arrow in range(num_arrows)], dtype=torch.long, device=device)
    features[off_indices] = 1

    num_active_arrows = 0
    tracking_index = 0

    # determine no. of active arrows
    for num_active in range(num_arrows + 1):
        n_steps = math.comb(num_arrows, num_active) * (NUM_ACTIVE_STATES ** num_active)

        if index < tracking_index + n_steps:
            num_active_arrows = num_active
            break
        else:
            tracking_index += n_steps

    if num_active_arrows > 0:
        # determine which arrows are active
        active_indices = []

        states_per_arrangement = NUM_ACTIVE_STATES ** num_active_arrows

        # all steps with first possible index enumerated first
        for arrow_idx in range(num_arrows):
            # find number of arrangements for steps starting w/current index
            n_arrangements = calc_arrangements(arrow_idx, num_active_arrows - len(active_indices), num_arrows - 1)
            if index < tracking_index + (n_arrangements * states_per_arrangement):
                active_indices.append(arrow_idx)
            else:
                tracking_index += n_arrangements * states_per_arrangement

            if len(active_indices) == num_active_arrows:
                break

        # determine the states of each arrow
        arrow_states = []
        for a in range(num_active_arrows):
            for state in range(1, NUM_ARROW_STATES):
                n_state_arrangements = (NUM_ACTIVE_STATES ** (num_active_arrows - a - 1))
                if index < tracking_index + n_state_arrangements:
                    arrow_states.append(state)
                    break
                else:
                    tracking_index += n_state_arrangements

        for idx, state in zip(active_indices, arrow_states):
            features[(idx * NUM_ARROW_STATES)] = 0
            features[(idx * NUM_ARROW_STATES) + state] = 1

    return features

def calc_arrangements(starting_index, num_indices, max_index):
    """ return the number of arrangements of increasing indices from starting_index -> max_index"""
    if num_indices == 1:
        return 1 # only the singleton arrangement [starting_index]
    else:
        return sum([calc_arrangements(next_index, num_indices - 1, max_index) 
                    for next_index in range(starting_index + 1, max_index - (num_indices - 1) + 2)])

def get_state_indices(arrow_idx, arrow_states, chart_type):
    """
    return all the vocabulary indices for states in which the arrow at arrow_idx has any
    of the given arrow_state(s)
    """
    num_arrows = (SELECTION_INPUT_SIZES[chart_type] - TIME_FEATURES) // NUM_ARROW_STATES
    vocab_size = SELECTION_VOCAB_SIZES[chart_type]
    step_states = set()

    other_idxs = [i for i in range(num_arrows) if i != arrow_idx]
    max_active = MAX_ACTIVE_ARROWS[chart_type]

    for state in arrow_states:
        note = ['.'] * num_arrows
        note[arrow_idx] = STATE_TO_UCS[state]	

        num_other_repeats = max_active if state == 0 else max_active - 1

        # possible states/permutations of the other indices
        other_states = list(itertools.product(STATE_TO_UCS.values(), repeat=num_other_repeats))
        other_idx_combinations = list(itertools.combinations(other_idxs, r=num_other_repeats))

        for other_state in other_states:
            for combination in other_idx_combinations:
                for i, step in enumerate(other_state):
                    note[combination[i]] = step
                
                step_states.add(''.join(note))

                for i, step in enumerate(other_state):
                    note[combination[i]] = '.'

    step_tensors = sequence_to_tensor(list(step_states))
    step_indices, _ = step_sequence_to_targets(step_tensors, chart_type, {})

    step_indices = [index.item() for index in step_indices]

    step_indices = list(filter(lambda x: x < vocab_size, step_indices))

    return step_indices
