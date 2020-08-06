# various constants/hyperparameters

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

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

SELECTION_VOCAB_SIZES = {
    'pump-single': 1,
    'pump-double': 1,
}
SELECTION_HIDDEN_WEIGHT = 0.7

NUM_SELECTION_LSTM_LAYERS = 2