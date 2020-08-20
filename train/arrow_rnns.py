# CLSTM placement model and LSTM RNN selection model

# model performing step placement; C-LSTM architecture as presented in
# dance dance convolution:  https://arxiv.org/abs/1703.06891 (section 4.2)

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# CNN part of the model takes in raw audio features and outputs processed features
class PlacementCNN(nn.Module):
    # params define the two convolution layers
    def __init__(self, in_channels, num_filters, kernel_sizes, pool_kernel, pool_stride, device):
        super().__init__()

        # maintain H/W dimensions (assume stride = dilation = 1)
        # ex) h_out = h_in + (2 * h_padding) - (1 * (kernel_size[0] - 1) - 1) + 1
        # wanted: h_out = h_in, -> h_padding = (-1 + (kernel_size[0] + 2))/2 = (kernel_size[0] + 1)/2  
        conv_padding = [((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
                        for kernel_size in kernel_sizes]
        
        self.conv1 = nn.Conv2d(in_channels=in_channels[0], out_channels=num_filters[0],
            kernel_size = kernel_sizes[0], padding=conv_padding[0])
        self.conv2 = nn.Conv2d(in_channels=in_channels[1], out_channels=num_filters[1],
            kernel_size = kernel_sizes[1], padding=conv_padding[1])

        # after each convLayer; maxPool2d only in frequency dim. -> ~ maxPool1d
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    # audio_input: [batch, 3, unroll_len, 80] (batch, channels, timestep, freq. bands)
    def forward(self, audio_input):
        timesteps = audio_input.size(2)     # 'h_in'
        freq_bands = audio_input.size(3)    # 'w_in'

        # [batch, num_filters, unroll_length, ?]
        conved1 = self.relu(self.conv1(audio_input))
        pooled1 = self.max_pool2d(conved1)

        # [batch, num_filters[-1], unroll_length, 160]
        conved2 = self.relu(self.conv2(pooled1))
        pooled2 = self.max_pool2d(conved2)

        # -> [batch, unroll_length, 160]; transpose, then flatten channel/freq. dimensions
        # shape is now (batch, timestep, processed features)
        result = pooled2.transpose(1, 2).flatten(2, 3)

        return result

# RNN + MLP part of the model (2nd half); take in processed audio features + chart type/level
class PlacementRNN(nn.Module):
    def __init__(self, num_lstm_layers, input_size, hidden_size, dropout=0.5):
        super().__init__()

        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        # dropout not applied to output of last lstm layer (need to apply manually)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)   # 0 or 1 for no step/step
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # processed_audio_input: [batch, timestep, input_size] (output of PlacementCNN.forward())
    #   timestep ~ unrolling length
    # chart_features: [batch, num_features] (concat. of one-hot representations)
    def forward(self, processed_audio_input, chart_features, states, input_lengths):
        batch_size = processed_audio_input.size(0)
        unroll_length = processed_audio_input.size(1)
        device = processed_audio_input.device

        # [batch, unroll_length, input_size] concat audio input with chart features
        #if all(input_lengths[0] == length for length in input_lengths):
        chart_features = chart_features.repeat(1, unroll_length).view(batch_size, unroll_length, -1)
        lstm_input = torch.cat((processed_audio_input, chart_features), dim=-1)

        # pack the sequence
        lstm_input_packed = pack_padded_sequence(lstm_input, input_lengths, batch_first=True, enforce_sorted=False)

        # [batch, unroll_length, hidden_size] (lstm_out: hidden states from last layer)
        # [2, batch, hidden] (hn/cn: final hidden cell states for both layers)
        lstm_out_packed, (hn, cn) = self.lstm(lstm_input_packed, states)

        # unpack output
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        
        # manual dropout to last lstm layer output
        lstm_out = self.dropout(lstm_out)

        # [batch, unroll_length, hidden] -> [batch, unroll, 128] -> [batch, unroll, 2] 
        # use last layer hidden states as input; 2 fully-connected relu layers w/dropout
        linear1_out = self.dropout(self.relu(self.linear1(lstm_out)))
        logits = self.dropout(self.relu(self.linear2(linear1_out)))

        # return logits directly, along hidden state for each frame
        # [batch, unroll_length, 2] / [batch, unroll, hidden] / ([batch, hidden] x 2)
        return logits, lstm_out.detach(), (hn.detach(), cn.detach())

    # initial celll/hidden state for lstm
    def initStates(self, batch_size, device):
        return (torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device))

# model responsible for step selection
# LSTM RNN architecture based off https://arxiv.org/abs/1703.06891 (section 4.4)
# with addition of an attention-inspired mechanism - utilize hidden states from
# placement model from the frame at which a particular step occurred

# In particular, at timestep t, construct a weighted sum of the resulting hidden state from
# the previous timestep, h_{t-1}, and the corresponding output from the clstm h_t':
# Note that not all hidden states from the placement model will be used.

# intuitively, stepcharters / players learn that certain step patterns tend to
# coincide with particular audio segments (i.e. jumps/brackets ~ accents, 
# drills ~ alternating notes, jacks ~ repeated notes, etc...) The above addition
# provides a way for the step selection algorithm to 'pay some attention' to the
# audio features at that particular frame via the appropriate clstm hidden state,
# *in addition* to the previous steps in the chart it has seen via the rnn hidden state

class SelectionRNN(nn.Module):
    def __init__(self, num_lstm_layers, input_size, output_size, hidden_size, hidden_weight, dropout = 0.5):
        super().__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        # how much to pay attention to hidden state from this rnn vs. from the placement model
        self.hidden_weight = hidden_weight
        self.placement_weight = 1 - hidden_weight

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # step_input: [batch, unroll_length, input_size]
    # clstm_hidden: [batch, hidden_size]
    # hidden, cell: [num_lstm_layers, batch, hidden_size]
    def forward(self, step_input, clstm_hidden, hidden, cell, input_lengths):
        # [2, batch, hidden] hidden_input at time t = a * h_{t-1} + b * h'_t', a is hidden_weight
        weighted_hidden = self.hidden_weight * hidden + (self.placement_weight *
            clstm_hidden.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1))

        lstm_input_packed = pack_padded_sequence(step_input, input_lengths, batch_first=True, enforce_sorted=False)

        # [batch, unroll_length, hidden] (lstm_out: hidden states from last layer)
        # [2, batch, hidden] (hn/cn: final hidden cell states for both layers)
        lstm_out_packed, (hn, cn) = self.lstm(step_input, (weighted_hidden, cell))

        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # manual dropout to last lstm layer output
        lstm_out = self.dropout(lstm_out)

        # [batch, unroll, output_size] - return logits directly
        linear1_out = self.dropout(self.relu(self.linear1(lstm_out)))
        logits = self.dropout(self.relu(self.linear2(linear1_out)))

        return logits, (hn.detach(), cn.detach())

    def initStates(self, batch_size, device):
        return (torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device))    
