import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharRNN(nn.Module):

    def __init__(self, input_size, hidden_units, output_size, embedding_size, num_layers=2):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units).to(device)
        
        x = self.embedding(x)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]

        logits = self.fc(out)

        return logits
