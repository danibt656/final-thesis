import random
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# We can use an image folder dataset the way we have it setup.
# Root directory for dataset
dataroot = "../data/PlantVillage/train"
# Create the dataset
image_dataset = dset.ImageFolder(root=dataroot,
                        transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
class_to_idx = image_dataset.class_to_idx
class_min_name = 'tomato_infected'
class_min = class_to_idx[class_min_name]
class_min_idxs = torch.nonzero(torch.Tensor(image_dataset.targets)==class_min).flatten()
image_dataset = Subset(image_dataset, class_min_idxs)

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

# Decide which device we want to run on
torch.cuda.empty_cache()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, relu='normal'):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if relu=='normal': self.relu = nn.ReLU()
        if relu=='leaky': self.relu = nn.LeakyReLU()

    def forward(self, x):
        sc = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + sc
        return self.relu(x)

class Generator(nn.Module):
    def __init__(self, in_channels=nz, out_channels=image_size, ngpu=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        #self.main = nn.Sequential(
        #    # input is Z, going into a convolution
        #    nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
        #    nn.BatchNorm2d(ngf * 8),
        #    nn.ReLU(True),
        #    # state size. ``(ngf*8) x 4 x 4``
        #    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf * 4),
        #    nn.ReLU(True),
        #    # state size. ``(ngf*4) x 8 x 8``
        #    nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf * 2),
        #    nn.ReLU(True),
        #    # state size. ``(ngf*2) x 16 x 16``
        #    nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf),
        #    nn.ReLU(True),
        #    # state size. ``(ngf) x 32 x 32``
        #    nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#       #      nn.ConvTranspose2d(ngf, nc, 10, 8, 1, bias=False), # imsize=256
        #    nn.Tanh()
        #    # state size. ``(nc) x 64 x 64``
        #)
        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False),
            ResBlock(64, 128, downsample=True),
            ResBlock(128, 128, downsample=False),
            ResBlock(128, 128, downsample=False),
            ResBlock(128, 64, downsample=True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(out_channels)
    
    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.gap(input)
        return nn.Tanh()(input)
        #return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, in_channels=nc, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        #discNorm = nn.BatchNorm2d
        #discNorm = nn.InstanceNorm2d
        #self.main = nn.Sequential(
        #    # input is ``(nc) x 64 x 64``
        #    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. ``(ndf) x 32 x 32``
        #    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #    discNorm(ndf * 2),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. ``(ndf*2) x 16 x 16``
        #    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #    discNorm(ndf * 4),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. ``(ndf*4) x 8 x 8``
        #    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #    discNorm(ndf * 8),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. ``(ndf*8) x 4 x 4``
        #    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#       #      nn.Conv2d(ndf * 8, 1, 3, 16, 0, bias=False), # imsize=256
        #    nn.Sigmoid() # In Wasserstein this is NOT between 0,1
        #    
        #)
        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResBlock(64, 64, downsample=False, relu='leaky'),
            ResBlock(64, 64, downsample=False, relu='leaky'),
            ResBlock(64, 64, downsample=False, relu='leaky'),
            ResBlock(64, 64, downsample=False, relu='leaky'),
            ResBlock(64, 64, downsample=False, relu='leaky'),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        # out_shape = (batch_size,1,1,1)
    
    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.gap(input)
        return nn.Sigmoid()(input)
        #return self.main(input)

# Create the generator
netG = Generator(in_channels=nz, out_channels=image_size, ngpu=ngpu).to(device)
netG.apply(weights_init)
noise = torch.randn(batch_size, nz, 1, 1).to(device)
fake = netG(noise)
print('done generator!')
netD = Discriminator(in_channels=nc, ngpu=ngpu).to(device)
netD.apply(weights_init)
# img, _ = next(iter(dataloader))
outd = netD(fake)
exit(1)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)
    
# Create the Discriminator
netD = Discriminator(ngpu=ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


img_list = []
G_losses = []
D_losses = []
iters = 0
# Number of training epochs
num_epochs = 400
CRITIC_ITERATIONS = 5

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # In Wasserstein: maximize D(x) - D(G(z))
        ###########################
#         for _ in range(CRITIC_ITERATIONS):
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
#             errD_real = -torch.mean(output) -torch.mean(label) # WGAN
        # Calculate gradients for D in backward pass
        errD_real.backward() # DCGAN
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batchreal_cpu
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward() # DCGAN
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake # DCGAN
#             errD = -torch.mean(errD_real) -torch.mean(errD_fake) # WGAN
#             errD.backward() # WGAN
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        # In Wasserstein: maximize D(G(z))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label) # DCGAN
#         errG = -torch.mean(output) # WGAN
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)) or iters==0:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('DGANloss.png')

torch.save(netG, 'saves/netG_PV.pt')
torch.save(netD, 'saves/netD_PV.pt')