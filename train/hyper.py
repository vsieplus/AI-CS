# various constants/hyperparameters

import math
from torch.nn import CrossEntropyLoss

SEED = 1950

PAD_IDX = -1

N_CHART_TYPES = 2
N_LEVELS = 10          # 28 levels -> 11 different ranges
CHART_FRAME_RATE = 100 # 10 ms per (chart) frame

BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_EPOCHS = 25
MAX_GRAD_NORM = 5

N_MELS = 80
N_FFTS = [1024, 2048, 4096]
HOP_LENGTH = 512

# actually bce loss, but use crossentropy for ignore_index functionality
PLACEMENT_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
PLACEMENT_LR = 0.001

PLACEMENT_AUDIO_PAD = 7  # how many frames of audio context to use during placement training
PLACEMENT_CHANNELS = [3, 10]
PLACEMENT_FILTERS = [10, 20]
PLACEMENT_KERNEL_SIZES = [(7, 3), (3, 3)]   # (time, frequency) [height, width]

PLACEMENT_POOL_STRIDE = (1, 3)
PLACEMENT_POOL_KERNEL = (1, 3)

PLACEMENT_UNROLLING_LEN = 100

# Thresholds for peak picking
MIN_THRESHOLD = 0.15
MAX_THRESHOLD = 0.35
MAX_CHARTLEVEL = 28

# only used as starting default (will be optimized on valid set after training for each model)
PLACEMENT_THRESHOLDS = [MIN_THRESHOLD + ((level - 1) / (MAX_CHARTLEVEL - 1)) * (MAX_THRESHOLD - MIN_THRESHOLD) for level in range(1, MAX_CHARTLEVEL + 1)]
PLACEMENT_THRESHOLDS = {str(key + 1): val for key,val in enumerate(PLACEMENT_THRESHOLDS)}

# chart_feats + output of cnn -> last filter size * pooled audio feats -> 160
PLACEMENT_INPUT_SIZE = N_CHART_TYPES + N_LEVELS + 160
NUM_PLACEMENT_LSTM_LAYERS = 2

SELECTION_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
SELECTION_LR = 0.0005

SELECTION_HIDDEN_WEIGHT = 0.8
NUM_SELECTION_LSTM_LAYERS = 2
SELECTION_UNROLLING_LEN = 64

TIME_FEATURES = 2

# arrow states: 
#   0 - OFF,
#   1 - ON (regular step),
#   2 - hold start 
#   3 - hold
#   4 - hold release
NUM_ARROW_STATES = 5
NUM_ACTIVE_STATES = NUM_ARROW_STATES - 1

# vocabulary mapping (indexing by # of activated arrows + states L -> R)
# (see stepchart.py, step_sequence_to_targets())
#   0 < index < vocab-size - 1, 
SELECTION_VOCAB_SIZES = {
    # exclude steps with 4+ active arrows at a time
    'pump-single': NUM_ARROW_STATES ** 5 - sum([math.comb(5, i) * (NUM_ACTIVE_STATES ** i) for i in range(4, 6)]),
    
    # 5 ^ 10 - SUM_(i=5->10) [(10 choose i) * (4 ^ i)] ~ ...
    'pump-double': NUM_ARROW_STATES ** 10 - sum([math.comb(10, i) * (NUM_ACTIVE_STATES ** i) for i in range(5, 11)]),
} 

SELECTION_INPUT_SIZES = {
    'pump-single': NUM_ARROW_STATES * 5 + TIME_FEATURES,    #  NUM_ARROW_STATES (which state) per arrow
    'pump-double': NUM_ARROW_STATES * 10 + TIME_FEATURES,   #  ^^^^
}

MAX_ACTIVE_ARROWS = {
    'pump-single': 3,
    'pump-double': 4,
}

CHECKPOINT_SAVE = 'checkpoint.tar'
CLSTM_SAVE = 'clstm.bin'
SRNN_SAVE = 'srnn.bin'

SUMMARY_SAVE = 'summary.json'
SPECIAL_TOKENS_SAVE = 'special_tokens.json'
THRESHOLDS_SAVE = 'placement_thresholds.json'
