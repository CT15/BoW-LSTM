import torch
import torch.nn as nn

class BowLSTM(nn.Module):
    def __init__(self, criterion, input_dim=1002, output_dim=1, hidden_dim=300, drop_prob=0.5):
        super(BowLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim =  output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=self.n_layers,
                            batch_first=True) # (batch, seq, feature)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.sigmoid = nn.Sigmoid() # nn.LeakyReLU(0.01)

        self.criterion = criterion


    def forward(self, x, X_lengths):
        # x of shape (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        X = nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True, enforce_sorted=False)
        X, hidden = self.lstm(X, self.init_hidden(batch_size))

        # X of shape (batch, seq_len, hidden_dim)
        X, length_list = nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=seq_len)

        # X of shape (batch * seq_len, hidden_dim)
        X = X.contiguous().view(-1, self.hidden_dim)

        # out of shape (batch * seq_len, output_size)
        X = self.dropout(X)
        X = self.fc(X)
        X = self.sigmoid(X)
        
        # X of shape (batch_size, seq_len * output_size = seq_len)
        X = X.view(batch_size, -1)
        
        return X, hidden


    def init_hidden(self, batch_size):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # hidden_a = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
        # hidden_b = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden
        # hidden_a = torch.autograd.Variable(hidden_a)
        # hidden_b = torch.autograd.Variable(hidden_b)
        # return (hidden_a, hidden_b)

    def loss(self, predictions, truths, lengths):
        assert predictions.size() == truths.size()
        
        batch_size, seq_len = predictions.size()

        # flatten all the labels
        truths = truths.view(-1)

        # flatten all the predictions
        predictions = predictions.view(-1)
        
        mask = (truths > -1).float()
        truths = truths * mask
        predictions = predictions * mask
        
        indices = []
        for i, length in enumerate(lengths):
            for j in range(length):
                indices.append(i * batch_size + j)

        truths = truths[indices]
        predictions = predictions[indices]

        loss = self.criterion.loss(predictions.float(), truths.float())

        return loss, torch.round(predictions), truths
