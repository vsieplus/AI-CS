# various constants/hyperparameters

from torch.nn import CrossEntropyLoss

SEED = 1949

PAD_IDX = -50

N_CHART_TYPES = 2
N_LEVELS = 28
AUDIO_FRAME_RATE = 100 # 10 ms per (audio) frame

BATCH_SIZE = 1
HIDDEN_SIZE = 128
NUM_EPOCHS = 12

# actually bce loss, but use crossentropy for ignore_index functionality
PLACEMENT_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
PLACEMENT_LR = 0.005
PLACEMENT_MAX_NORM = 5

PLACEMENT_AUDIO_PAD = 7  # how many frames of audio context to use during placement training
PLACEMENT_CHANNELS = [3, 10]
PLACEMENT_FILTERS = [10, 20]
PLACEMENT_KERNEL_SIZES = [(7, 3), (3, 3)]   # (time, frequency) [height, width]

PLACEMENT_POOL_STRIDE = (1, 3)
PLACEMENT_POOL_KERNEL = (1, 3)

PLACEMENT_UNROLLING_LEN = 100       # 100 frames of unrolling for placement model

# chart_feats + output of cnn -> last filter size * pooled audio feats -> 160
PLACEMENT_INPUT_SIZE = N_CHART_TYPES + N_LEVELS + 160
NUM_PLACEMENT_LSTM_LAYERS = 2

SELECTION_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
SELECTION_LR = 0.005

# arrow states: 
#   0 - OFF,
#   1 - ON (regular step or hold start), 
#   2 - Hold, 
#   3 - RELEASE (hold)
NUM_ARROW_STATES = 4

# vocabulary mapping (base 4 L->R)
#   step -> features (dim 20) -> index  [assume indexing start at 1]
#   step_index = SUM(i=0->4)[step[i] * (4^i)]
#   ex) '01021' -> 0 + (1 * 4) + 0 + (2 * 4^3) + (1 * 4 ^ 4) 
#   0 < index < vocab-size - 1, 
SELECTION_VOCAB_SIZES = {
    'pump-single': NUM_ARROW_STATES ** 5,   # 1024 possible states
    'pump-double': NUM_ARROW_STATES ** 10,  # 1,048,576 (if unbounded) -> limit?
}

SELECTION_HIDDEN_WEIGHT = 0.7

NUM_SELECTION_LSTM_LAYERS = 2