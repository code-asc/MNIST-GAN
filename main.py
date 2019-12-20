import torch
import numpy as np
import pickle as pkl
from torchvision import datasets
from torchvision import transforms
from generator import Generator
from discriminator import Discriminator
import torch.backends.cudnn as cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def realLoss(D_out):
    b_size = D_out.size(0)
    labels = torch.ones(b_size) * 0.9
    labels = labels.to(device)
    loss_fun = torch.nn.BCEWithLogitsLoss()
    loss = loss_fun(D_out.squeeze(), labels)
    return loss

def fakeLoss(D_out):
    b_size = D_out.size(0)
    labels = torch.zeros(b_size)
    labels = labels.to(device)
    loss_fun = torch.nn.BCEWithLogitsLoss()
    loss = loss_fun(D_out.squeeze(), labels)
    return loss

BATCH_SIZE = 128
PRINT_INTERVAL = 500
EPOCH = 100

input_size = 784
d_output_size = 1
d_hidden_size = 32
latent_size = 100
g_output_size = 784
g_hidden_size = 32
lr = 0.002
num_samples = 16

transform = transforms.ToTensor()

trainset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE)

D = Discriminator(input_size, d_hidden_size, d_output_size)
G = Generator(latent_size, g_hidden_size, g_output_size)

D = D.to(device)
G = G.to(device)

D.train()
G.train()

if device == 'cuda':
    D= torch.nn.DataParallel(D)
    G = torch.nn.DataParallel(G)
    cudnn.benchmark = True

d_optimizer = torch.optim.Adam(D.parameters(), lr)
g_optimizer = torch.optim.Adam(G.parameters(), lr)

fixed_z = np.random.uniform(-1, 1, size=(num_samples, latent_size))
fixed_z = torch.from_numpy(fixed_z).float()


samples = []

for epoch in range(EPOCH):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        b_size = inputs.size(0)

        d_optimizer.zero_grad()

        #========================================
        #   Training Discriminator
        #   Discriminator loss on real and fake images
        #========================================
        d_real_pred = D(inputs)
        d_real_loss = realLoss(d_real_pred)

        z = np.random.uniform(-1, 1, size=(b_size, latent_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)

        d_fake_pred = D(fake_images)
        d_fake_loss = fakeLoss(d_fake_pred)

        d_total_loss = d_real_loss + d_fake_loss
        d_total_loss.backward()
        d_optimizer.step()

        #========================================
        #   Training Generator
        #========================================
        g_optimizer.zero_grad()
        z = np.random.uniform(-1, 1, size=(b_size, latent_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)

        d_fake_pred = D(fake_images)
        g_loss = realLoss(d_fake_pred)

        g_loss.backward()
        g_optimizer.step()

        if batch_idx % PRINT_INTERVAL == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_total_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch+1, EPOCH, d_total_loss.item(), g_loss.item()))

    samples_z = G(fixed_z)
    samples.append(samples_z)

with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

torch.save(model.state_dict(), './MNIST_state.mdl')
print('Model state saved....')
torch.save(model, './MNIST_model.mdl')
print('Model saved....')
