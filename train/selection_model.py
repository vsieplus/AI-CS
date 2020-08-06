# model responsible for step selection
# LSTM RNN architecture based off https://arxiv.org/abs/1703.06891 (section 4.4)
# Addition - utilize hidden states from placement model from the frame at which
# a particular step occurred. 

# In particular, at timestep t, construct a weighted sum of the resulting hidden state from
# the previous timestep, h_{t-1}, and the corresponding output from the placement model RNN h_t':
# Note that not all hidden states from the placement model will be used.

import torch
import torch.nn as nn

class SelectionRNN(nn.Module):
    def __init__(self, num_lstm_layers, input_size, hidden_size, hidden_weight, dropout):
        super(SelectionRNN, self).__init__()
        
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.hidden_weight = hidden_weight
        self.placement_weight = 1 - hidden_weight

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_lstm_layers, dropout=dropout, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # step_input: [batch, unfoldings, input_size]
    # placement_hidden, hidden, cell: [num_lstm_layers, batch, hidden_size]
    def forward(self, step_input, placement_hidden, hidden, cell):
        # [batch, hidden] hidden_input at time t = a * h_{t-1} + b * h_t', a is hidden_weight
        weighted_hidden = self.hidden_weight * hidden + self.placement_weight * placement_hidden

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