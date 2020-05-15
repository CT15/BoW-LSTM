import torch
import torch.nn as nn

class BowLSTM(nn.Module):
    """
    Despite the class name, this class is actually a general single LSTM
    layer that is followed by a fully connected layer at every time step.

    ...
    Attributes
    ----------
    criterion:
        loss function
    input_dim: int
        the number of expected features in the input (default 1002)
    output_dim: int
        the number of expected features in the output (default 1)
    hidden_dim:
        the number of expected features in the hidden state (default 300)
    drop_prob:
        probability of an element of the hidden state of the second LSTM layer to 
        be zeroed (default 0.5)
    lstm: torch.nn.LSTM
        LSTM layer
    dropout: torch.nn.Dropout
        dropout layer with p = drop_prob
    fc: torch.nn.Linear
        linear transformation that transforms hidden states to the output vectors
    sigmoid:
        sigmoid function applied to each element of the output tensor

    Methods
    -------
    forward(x, X_lengths)
        Computation performed at every call
    loss(predictions, truths, lengths)
        Returns the loss value of the predictions as compared to the truths

    """

    def __init__(self, criterion, input_dim=1002, output_dim=1, hidden_dim=300, drop_prob=0.5):
        """
        Parameters
        ----------
        criterion:
            loss function
        input_dim : int
            the number of expected features in the input (default 1002)
        output_dim : int
            the number of expected features in the output (default 1)
        hidden_dim : int
            the number of expected features in the hidden state (default 300)
        drop_prob: float
            probability of an element of the hidden state of the second LSTM layer to 
            be zeroed (default 0.5)
        """

        super(BowLSTM, self).__init__()
        self.input_dim = input_dim
        self.output_dim =  output_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True) # (batch, seq, feature)

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.sigmoid = nn.Sigmoid() # nn.LeakyReLU(0.01)

        self.criterion = criterion


    def forward(self, x, X_lengths):
        """Computation performed at every call

        Parameters
        ----------
        x: input tensors
        X_lengths: the number of posts in each thread
    
        Input Shape
        -----------
        x: (batch, seq_len, input_size)
        X_lengths: (batch, seq_len)
        

        Output Shape
        ------------
        X: (batch, seq_len, output_dim)
        hidden: (h_n, c_n) refer to pytorch doc 

        """

        batch_size, seq_len, _ = x.size()
        
        X = nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True, enforce_sorted=False)
        X, hidden = self.lstm(X, self._init_hidden(batch_size))

        # X of shape (batch, seq_len, hidden_dim)
        X, length_list = nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=seq_len)

        # X of shape (batch * seq_len, hidden_dim)
        #X = X.contiguous().view(-1, self.hidden_dim)

        # X of shape (batch, seq_len, output_dim)
        X = self.dropout(X)
        X = self.fc(X)
        X = self.sigmoid(X)

        X = X.view(batch_size, -1)
        
        return X, hidden

    # I realise that this method is actually redundant given
    # that the initial hidden state is already 0 by default
    def _init_hidden(self, batch_size):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # hidden_a = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
        # hidden_b = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

        # the following implementation makes more sense (optional):
        # hidden_a = torch.autograd.Variable(hidden_a)
        # hidden_b = torch.autograd.Variable(hidden_b)
        # return (hidden_a, hidden_b)


    def loss(self, predictions, truths, lengths):
        """Returns the loss value of the predictions as compared to the truths

        For convenience, this method also returns all the predictions and truths.
        Predictions and truths are useful for various calculations, for example
        F1 score, precision, recall and confusion matrix.

        ...

        Parameters
        ----------
        predictions: predicted values
        truths: ground truth values
        lengths: the number of posts in each thread in the batch

        Input Shape
        -----------
        predictions: (batch_size, seq_len)
        truths: (batch_size, seq_len)
        posts_lengths: (batch_size)

        Output Shape
        ------------
        loss: float
        predictions: (batch_size * seq_len) at most (can be less)
        truths: (batch_size * seq_len) at most (can be less)

        """
    
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
