# baseline placement and selection models

from collections import Counter

import torch.nn as nn

# logreg (logistic regression) placement model
class PlacementLogReg(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.linear(x)
        return output

# mlp placement model
class PlacementMLP(nn.Module):
    def __init__(self, input_size, output_size, linear_1_out=256, linear_2_out=128, dropout=0.5):
        super().__init__()

        self.linear1 = nn.Linear(input_size, linear_1_out)
        self.linear2 = nn.Linear(linear_1_out, linear_2_out)
        self.linear_out = nn.Linear(linear_2_out, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out1 = self.dropout(self.relu(self.linear1(x)))
        out2 = self.dropout(self.relu(self.linear2(out1)))
        
        out = self.linear_out(out2)

        return out

# smoothed n-gram selection models
# adapted from https://github.com/chrisdonahue/ddc/blob/master/learn/ngram.py
class SelectionNGram:
    def __init__(self, ngram_counts, n=5):
        """
        ngram_counts is a Counter object with counts of (str representations) ngrams
        """
        self.n = n
        self.ngram_total = sum(ngram_counts.values())

        self.history_counts = Counter()
        for ngram, count in ngram_counts.items():
            self.history_counts[ngram[:-1]] += count

        self.vocab = set()
        for ngram, _ in ngram_counts.items():
            for w in ngram:
                self.vocab.add(w)

    def mle(self, ngram):
        ngram_count = self.ngram_counts[ngram]          # how many x the ngram appears
        history_count = self.history_counts[ngram[:-1]] # how many times the first n-1 tokens appear
        return ngram_count / history_count

    # laplace smoothing
    def laplace(self, ngram, smoothing=1):
        history_count = self.history_counts[ngram[:-1]]
        numerator = smoothing + self.ngram_counts[ngram]
        denominator = len(self.vocab) + smoothing + history_count

        return numerator / denominator

# fixed-window 5 gram selection mlp
class SelectionNGramMLP(nn.Module):
    def __init__(self, vocab_size, input_size, linear_1_out=256, linear_2_out=128, n=5):
        super().__init__()

        self.linear1 = nn.Linear((n - 1) * input_size, linear_1_out)

        self.linear2 = nn.Linear(linear_1_out, linear_2_out)
        self.linear_out = nn.Linear(linear_2_out, vocab_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [n-1, input_size]
        x = x.view(1, -1)

        out1 = self.dropout(self.relu(self.linear1(x)))
        out2 = self.dropout(self.relu(self.linear2(out1)))
        
        out = self.linear_out(out2)

        return out