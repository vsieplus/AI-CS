# various constants/hyperparameters

import math
from torch.nn import CrossEntropyLoss

SEED = 1949

PAD_IDX = -1

N_CHART_TYPES = 2
N_LEVELS = 28
CHART_FRAME_RATE = 100 # 10 ms per (chart) frame

BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_EPOCHS = 25
MAX_GRAD_NORM = 5

HOP_LENGTH = 512

# actually bce loss, but use crossentropy for ignore_index functionality
PLACEMENT_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
PLACEMENT_LR = 0.0001

PLACEMENT_AUDIO_PAD = 7  # how many frames of audio context to use during placement training
PLACEMENT_CHANNELS = [3, 10]
PLACEMENT_FILTERS = [10, 20]
PLACEMENT_KERNEL_SIZES = [(7, 3), (3, 3)]   # (time, frequency) [height, width]

PLACEMENT_POOL_STRIDE = (1, 3)
PLACEMENT_POOL_KERNEL = (1, 3)

PLACEMENT_UNROLLING_LEN = 100

# Thresholds for peak picking
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.75
MAX_CHARTLEVEL = 28

# chart_feats + output of cnn -> last filter size * pooled audio feats -> 160
PLACEMENT_INPUT_SIZE = N_CHART_TYPES + N_LEVELS + 160
NUM_PLACEMENT_LSTM_LAYERS = 2

SELECTION_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
SELECTION_LR = 0.001

SELECTION_HIDDEN_WEIGHT = 0.8
NUM_SELECTION_LSTM_LAYERS = 2
SELECTION_UNROLLING_LEN = 64

# arrow states: 
#   0 - OFF,
#   1 - ON (regular step or hold start), 
#   2 - Hold, 
#   3 - RELEASE (hold)
NUM_ARROW_STATES = 4

# vocabulary mapping (indexing by # of activated arrows + states L -> R)
# (see stepchart.py, step_sequence_to_targets())
#   0 < index < vocab-size - 1, 
SELECTION_VOCAB_SIZES = {
    'pump-single': NUM_ARROW_STATES ** 5,   # 1024 possible states
    
    # 1,048,576 - SUM_(i=5->10) [(10 choose i) * (3 ^ i)] ~ 20,686
    'pump-double': NUM_ARROW_STATES ** 10 - sum([math.comb(10, i) * (3 ** i) for i in range(5, 11)]),
} 

SELECTION_INPUT_SIZES = {
    'pump-single': NUM_ARROW_STATES * 5,    # 20 element vector / 4 per arrow
    'pump-double': NUM_ARROW_STATES * 10,   # 40 element vector / ^^^^
}

MAX_ACTIVE_ARROWS = {
    'pump-single': 5,
    'pump-double': 4
}

CHECKPOINT_SAVE = 'checkpoint.tar'
CLSTM_SAVE = 'clstm.bin'
SRNN_SAVE = 'srnn.bin'
