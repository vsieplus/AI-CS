# various constants/hyperparameters

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

PAD_IDX = -5

BATCH_SIZE = 1
HIDDEN_SIZE = 256
NUM_EPOCHS = 5

PLACEMENT_CRITERION = BCEWithLogitsLoss()
PLACEMENT_LR = 0.005

PLACEMENT_CHANNELS = [3, 10]
PLACEMENT_FILTERS = [10, 20]
PLACEMENT_KERNEL_SIZES = [(7, 3), (3, 3)]

NUM_PLACEMENT_LSTM_LAYERS = 2
NUM_PLACEMENT_FEATURES = 30 # 2 chart types, 28 difficulty levels

SELECTION_CRITERION = CrossEntropyLoss()
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