# various constants/hyperparameters

from torch.nn import CrossEntropyLoss

SEED = 1949

PAD_IDX = -50

N_CHART_TYPES = 2
N_LEVELS = 28
N_AUDIO_FEATS = 83     # 80 frequency bands + 3 channels
AUDIO_FRAME_RATE = 100 # 10 ms per (audio) frame

BATCH_SIZE = 1
HIDDEN_SIZE = 128
NUM_EPOCHS = 5

PLACEMENT_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
PLACEMENT_LR = 0.005
PLACEMENT_MAX_NORM = 5

PLACEMENT_AUDIO_PAD = 7  # how many frames of audio context to use during placement training
PLACEMENT_CHANNELS = 3
PLACEMENT_FILTERS = [10, 20]
PLACEMENT_KERNEL_SIZES = [(7, 3), (3, 3)]   # (time, frequency) [height, width]

PLACEMENT_POOL_STRIDE = (1, 3)
PLACEMENT_POOL_WIDTH = (1, 3)

PLACEMENT_UNROLLINGS_LEN = 100       # 100 frames of unrolling for placement model

PLACEMENT_INPUT_SIZE = N_CHART_TYPES + N_LEVELS + N_AUDIO_FEATS
NUM_PLACEMENT_LSTM_LAYERS = 2

SELECTION_CRITERION = CrossEntropyLoss(ignore_index=PAD_IDX)
SELECTION_LR = 0.005

# arrow states: 
#   0 - OFF,
#   1 - ON (regular step or hold start), 
#   2 - Hold, 
#   3 - RELEASE (hold)
NUM_ARROW_STATES = 5 

# vocabulary mapping
#   index -> feature_representation -> step
#   241 -> [0, 1, 0, 0, 1, 0, ..., 0, 1] -> '10012'
#   0 < index < vocab-size - 1, 
SELECTION_VOCAB_SIZES = {
    'pump-single': NUM_ARROW_STATES ** 5,   # 1024 possible states
    'pump-double': NUM_ARROW_STATES ** 10,  # 1,048,576 (if unbounded) -> limit?
}

SELECTION_HIDDEN_WEIGHT = 0.7

NUM_SELECTION_LSTM_LAYERS = 2