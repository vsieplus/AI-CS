# model performing step placement; C-LSTM architecture as presented in
# https://arxiv.org/abs/1703.06891 (section 4.2)

import torch
import torch.nn as nn

# CNN part of the model takes in raw audio features (first half)
class PlacementCNN(nn.Module):
    # params define the two convolution layers
    def __init__(self, in_channels, num_filters, kernel_sizes):
        
        super(PlacementCNN, self).__init__()

        # conv. layer filter sizes [7, 3]/[3, 3] ~ [time, frequency] as H/W dims
        self.convLayer1 = nn.Conv2d(in_channels=in_channels[0],
            out_channels=num_filters[0], kernel_size=kernel_sizes[0])
        self.convLayer2 = nn.Conv2d(in_channels=in_channels[1],
            out_channels=num_filters[1], kernel_size=kernel_sizes[1])

        # after each convLayer; maxPool2d only in frequency dim. -> ~ maxPool1d
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
        # shape is now (batch, timestep, features)
        result = result.transpose(2, 3).flatten(2, 3)

        return result

# RNN + MLP part of the model (2nd half); take in processed audio features + chart features
class PlacementRNN(nn.Module):
    def __init__(self, num_lstm_layers, num_features, hidden_size, dropout=0.5):
        super(PlacementRNN, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        # dropout not applied to output of last lstm layer (need to apply manually)
        self.lstm = nn.LSTM(input_size=28 + num_features, hidden_size=hidden_size,
            num_layers=num_lstm_layers, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # processed_audio_input: [batch, 7, 28] (output of PlacementCNN.forward())
    # chart_features: [batch, num_features] (concat. of one-hot representations)
    def forward(self, processed_audio_input, chart_features):
        batch_size = processed_audio_input.size(0)

        # [batch, 7, 28 + num_features] concat audio input with chart features
        chart_features = chart_features.repeat(1, 7).view(batch_size, 7, -1)
        lstm_input = torch.cat((processed_audio_input, chart_features), dim=-1)

        # [batch, 7, hidden] (output: hidden states from last layer)
        # [2, batch, hidden] (hn/cn: final hidden cell states for both layers)
        lstm_out, (hn, cn) = self.lstm(lstm_input, initStates(batch_size, device)))
        
        # manual dropout to last lstm layer output
        lstm_out = self.dropout(lstm_out)

        # [batch, hidden] (use last hidden states as input to linear layer)
        linear_input = lstm_output[:, -1]

        # -> [batch, 128] -> [batch, 1] (2 fully-connected relu layers w/dropout)
        linear_input = self.dropout(self.relu(self.linear1(linear_input)))
        linear_output = self.dropout(self.relu(self.linear1(linear_input)))

        # [batch, 1] (return logits directly)
        return linear_output

    # initial celll/hidden state for lstm
    def initStates(self, batch_size, device):
        return (torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device))