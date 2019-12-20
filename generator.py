import torch
import torch.nn.functional as F

class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc4 = torch.nn.Linear(hidden_dim * 4, output_size)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)

        out = F.tanh(self.fc4(x))
        return out
