import torch
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = torch.nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, output_size)

        self.dropout = torch.nn.Dropout(0.3)


    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)

        out = self.fc4(x)
        return out
