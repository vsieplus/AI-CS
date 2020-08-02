# model performing step placement; C-LSTM architecture as presented in
# https://arxiv.org/abs/1703.06891 (section 4.2)

import torch
import torch.nn as nn

# CNN part of the model takes in raw audio features (first half)
class PlacementCNN(nn.Module):
    def __init__(self, ):
        super(PlacementCNN, self).__init__()

        self.hidden_size = hidden_size

        # conv. layer filter sizes [7, 3]/[3, 3] ~ [time, frequency] as Height/Width dimensions
        self.convLayer1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(7,3))
        self.convLayer2 = nn.Conv2d(in_channels=10 , out_channels=20, kernel_size=(3,3))

        # after each convLayer; maxPool2d only in frequency dim. -> equivalent to maxPool1d
        self.relu = nn.ReLU()
        self.maxPool1d = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))

    # audio_input: [batch, 3, 15, 80] (batch, channels, timestep, freq.)
    def forward(self, audio_input):
        # [batch, 10, 9, 78]
        result = self.relu(self.convLayer1(audio_input))

        # [batch, 10, 9, 26]
        result = self.maxPool1d(result)

        # [batch, 20, 7, 24]
        result = self.relu(self.convLayer2(result))

        # [batch, 20, 7, 8]; still (batch, channels, timestep, freq)
        result = self.maxPool1d(result)

        # [batch, 7, 28]; transpose, then flatten channel/freq. dimensions
        # shape is now (batch, time, features)
        result = result.transpose(2, 3).flatten(2, 3)

        return result

# RNN + MLP part of the model (2nd half); take in processed audio features + chart features
class PlacementRNN(nn.Module):
    def __init__(self, num_lstm_layers, hidden_size, num_levels, num_chart_types,
        dropout=0.5):
        
        super(PlacementRNN, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=28 + num_levels + num_chart_types, batch_first=True,
            hidden_size=hidden_size, num_layers=num_lstm_layers, dropout=dropout)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()

    # processed_audio_input: [batch, 7, 28] (output of PlacementCNN.forward())
    # diff_level_input: [batch, 28] (one-hot representation of chart level)
    # chart_type_input: [batch, 2] (one-hot representation of chart type)
    def forward(self, processed_audio_input, diff_level_input, chart_type_input):
        pass

    def initHidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device = device)

    def initCell(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device = device)