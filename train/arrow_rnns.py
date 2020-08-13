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
    def __init__(self, in_channels, num_filters, kernel_sizes, pool_kernel, pool_stride):
        super().__init__()

        # conv layers use [time, frequency] as H/W dims; use same padding to maintain dims        
        k = 1
        conv_weights = [torch.randn(n_filters, in_channel, kernel_size[0], kernel_size[1]) *
                        (-sqrt(k) - sqrt(k)) + sqrt(k) # uniform dist on [-1, 1]  
                        for n_filters, kernel_size, in_channel in zip(num_filters, kernel_sizes, in_channels)]

        # maintain H/W dimensions (assume stride = dilation = 1)
        # ex) h_out = h_in + (2 * h_padding) - (1 * (kernel_size[0] - 1) - 1) + 1
        # wanted: h_out = h_in, -> h_padding = (-1 + (kernel_size[0] + 2))/2 = (kernel_size[0] + 1)/2  
        conv_padding = [((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
                        for kernel_size in kernel_sizes]
                    
        self.conv_params = zip(conv_weights, conv_padding)

        # after each convLayer; maxPool2d only in frequency dim. -> ~ maxPool1d
        self.relu = nn.ReLU()
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

    # audio_input: [batch, 3, unroll_len, 80] (batch, channels, timestep, freq. bands)
    def forward(self, audio_input):
        timesteps = audio_input.size(2)     # 'h_in'
        freq_bands = audio_input.size(3)    # 'w_in'
        
        result = audio_input

        for conv_weight, conv_padding in self.conv_params:
            result = F.conv2d(result, weight=conv_weight.to(audio_input.device), 
                padding=conv_padding)

            # [batch, num_filters, unroll_length, ?]
            result = self.relu(result)

            # [batch, num_filters, unroll_length, ?]
            result = F.max_pool2d(result, kernel_size=self.pool_kernel, stride=self.pool_stride)

        # [batch, num_filters[-1], unroll_length, ?]
        # -> [batch, unroll_length, # features]; transpose, then flatten channel/freq. dimensions
        # shape is now (batch, timestep, features)
        result = result.transpose(1, 2).flatten(2, 3)

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
        self.linear2 = nn.Linear(in_features=128, out_features=2)   # 0 or 1 for step/no step
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # processed_audio_input: [batch, timestep, input_size] (output of PlacementCNN.forward())
    #   timestep ~ unrolling length
    # chart_features: [batch, num_features] (concat. of one-hot representations)
    def forward(self, processed_audio_input, chart_features, states, input_lengths):
        breakpoint()
        batch_size = processed_audio_input.size(0)
        unroll_length = processed_audio_input.size(1)
        device = processed_audio_input.device

        # [batch, unroll_length, input_size] concat audio input with chart features
        # only concatenate the features when no sequence is padded (leave out towards the end)
        if all(input_lengths[0] == length for length in input_lengths):
            chart_features = chart_features.repeat(1, unroll_length).view(batch_size, unroll_length, -1)
            lstm_input = torch.cat((processed_audio_input, chart_features), dim=-1)
        else:
            lstm_input = processed_audio_input

        # pack the sequence
        lstm_input = pack_padded_sequence(lstm_input, input_lengths, batch_first=True, enforce_sorted=False)

        # [batch, unroll_length, hidden_size] (lstm_out: hidden states from last layer)
        # [2, batch, hidden] (hn/cn: final hidden cell states for both layers)
        lstm_out, (hn, cn) = self.lstm(lstm_input, states)

        # unpack output
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # manual dropout to last lstm layer output
        lstm_out = self.dropout(lstm_out)

        # [batch, hidden] (use last layer hidden states as input to linear layer for each timestep)
        all_logits = []
        for i in range(unroll_length):
            # [batch, hidden] -> [batch, 128] -> [batch, 1] (2 fully-connected relu layers w/dropout)
            linear_input = lstm_out[:, i]
            linear_input = self.dropout(self.relu(self.linear1(linear_input)))
            logits = self.dropout(self.relu(self.linear2(linear_input)))

            all_logits.append(logits.unsqueeze(0))

        logits = torch.cat(all_logits, dim=0)
        all_logits.clear()

        # return logits directly, along hidden state for each frame
        # [batch, unroll_length, 2] / [batch, unroll, hidden] / ([batch, hidden] x 2)
        return logits, lstm_out.detach(), (hn, cn)

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
    def __init__(self, num_lstm_layers, input_size, hidden_size, hidden_weight, dropout = 0.5):
        super().__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size

        # how much to pay attention to hidden state from this rnn/the placement model
        self.hidden_weight = hidden_weight
        self.placement_weight = 1 - hidden_weight

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers, dropout=dropout, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # step_input: [batch, unfoldings, input_size]
    # clstm_hidden, hidden, cell: [num_lstm_layers, batch, hidden_size]
    def forward(self, step_input, clstm_hidden, hidden, cell):
        # [batch, hidden] hidden_input at time t = a * h_{t-1} + b * h_t', a is hidden_weight
        weighted_hidden = self.hidden_weight * hidden + self.placement_weight * clstm_hidden

        # [batch, unfoldings, hidden] (lstm_out: hidden states from last layer)
        # [2, batch, hidden] (hn/cn: final hidden cell states for both layers)
        lstm_out, (hn, cn) = self.lstm(step_input, (weighted_hidden, cell))

        # manual dropout to last lstm layer output
        lstm_out = self.dropout(lstm_out)

        # [batch, hidden] (use hidden states from last timestep as input to linear layer)
        linear_input = lstm_out[:, -1]

        # [batch, input_size] - return logits directly
        linear_out = self.dropout(self.relu(self.linear(linear_input)))

        return linear_out

    def initStates(self, batch_size, device):
        return (torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=device))    
