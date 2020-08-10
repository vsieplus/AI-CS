# model performing step placement; C-LSTM architecture as presented in
# https://arxiv.org/abs/1703.06891 (section 4.2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# CNN part of the model takes in raw audio features (first half)
class PlacementCNN(nn.Module):
    # params define the two convolution layers
    def __init__(self, in_channels, num_filters, kernel_sizes, pool_kernel, pool_stride):
        super(PlacementCNN, self).__init__()

        # conv layers use [time, frequency] as H/W dims; use same padding to maintain dims
        self.in_channels = in_channels
        self.conv_params = zip(num_filters, kernel_sizes)

        # after each convLayer; maxPool2d only in frequency dim. -> ~ maxPool1d
        self.relu = nn.ReLU()
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

    # audio_input: [batch, 3, ?, 80] (batch, channels, timestep, freq. bands)
    def forward(self, audio_input):
        timesteps = audio_input.size(2)     # 'h_in'
        freq_bands = audio_input.size(3)    # 'w_in'
        
        for num_filters, kernel_size in self.conv_params:
            # maintain H/W dimensions (assume stride = dilation = 1)
            # ex) h_out = h_in + (2 * h_padding) - (1 * (kernel_size[0] - 1) - 1) + 1
            # wanted: h_out = h_in, -> h_padding = (-1 + (kernel_size[0] + 2))/2 = (kernel_size[0] + 1)/2  
            # analgous for w_padding
            required_conv_padding = ((kernel_size[0] + 1) / 2, (kernel_size[1] + 1) / 2)
            
            result = F.conv2d(audio_input, weights=(num_filters, self.in_channels,
                kernel_size[0], kernel_size[1]), padding=required_conv_padding)

            # [batch, num_filters, ?, 80]
            result = self.relu(result)

            # also maintain h/w through max-pooling layer
            # h_out = ((h_in + (2 * h_padding) - (1 * (kernel_size[0] - 1) - 1)) / stride[0] ) + 1 
            # -> (h_in + (2 * h_padding) - (kernel_size[0] - 2)) = stride[0] * h_in - stride[0]
            # -> h_padding = ((stride[0] - 1) * h_in - stride[0] + kernel_size[0] - 2) / 2
            required_pool_padding = (((stride[0] - 1) * timesteps - stride[0] + kernel_size[0] - 2) / 2,
                ((stride[1] - 1) * freq_bands - stride[1] + kernel_size[1] - 2) / 2)

            # [batch, num_filters, ?, 80]
            result = F.max_pool2d(result, kernel_size=self.pool_kernel, stride=self.pool_stride,
                padding=required_pool_padding)

        # [batch, num_filters[1], ?, 80]
        # -> [batch, ?, 83]; transpose, then flatten channel/freq. dimensions
        # shape is now (batch, timestep, features)
        result = result.transpose(1, 2).flatten(2, 3)

        return result

# RNN + MLP part of the model (2nd half); take in processed audio features + chart type/level
class PlacementRNN(nn.Module):
    def __init__(self, num_lstm_layers, input_size, hidden_size, dropout=0.5):
        super(PlacementRNN, self).__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        # dropout not applied to output of last lstm layer (need to apply manually)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # processed_audio_input: [batch, timestep, input_size] (output of PlacementCNN.forward())
    #   timestep ~ unrolling length
    # chart_features: [batch, num_features] (concat. of one-hot representations)
    def forward(self, processed_audio_input, chart_features):
        batch_size = processed_audio_input.size(0)
        seq_len = processed_audio_input.size(1)

        # [batch, 7, 28 + num_features] concat audio input with chart features
        chart_features = chart_features.repeat(1, seq_len).view(batch_size, seq_len, -1)
        lstm_input = torch.cat((processed_audio_input, chart_features), dim=-1)

        # [batch, 7, hidden] (lstm_out: hidden states from last layer)
        # [2, batch, hidden] (hn/cn: final hidden cell states for both layers)
        lstm_out, (hn, cn) = self.lstm(lstm_input, initStates(batch_size, device)))
        
        # manual dropout to last lstm layer output
        lstm_out = self.dropout(lstm_out)

        # hidden state for current frame (potentially save for later use in selection model)
        curr_frame_hidden = lstm_out[:, int(lstm_out.size(1)/2)]

        # [batch, hidden] (use last hidden states as input to linear layer)
        linear_input = lstm_output[:, -1]

        # -> [batch, 128] -> [batch, 1] (2 fully-connected relu layers w/dropout)
        linear_input = self.dropout(self.relu(self.linear1(linear_input)))
        linear_output = self.dropout(self.relu(self.linear1(linear_input)))

        # [batch, 1] (return logits directly, along hidden state for curr. [middle] frame)
        return linear_output, curr_frame_hidden

    # initial celll/hidden state for lstm
    def initStates(self, batch_size, device):
        return (torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device))