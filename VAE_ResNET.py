"""Author: Zahra Sheikhbahaee- March 2020 """
import torch
from torch import nn, optim
import torch.nn.functional as F
import os
from torch.utils import data
from torchvision import datasets, transforms
from PIL import Image

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x

class VAE(nn.Module):

    def __init__(self, z_dim=128):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        """
            path
        """
        self.images = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
        self.transform = transform

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

def get_dataset(path, img_size, batch_size):
    #
    transform = transforms.Compose([
                  transforms.Resize(img_size),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                       std= (0.5, 0.5, 0.5))])

    datasets = ImageDataset(path, transform)
    data_loader = data.DataLoader(dataset=datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
    return data_loader


class Config(object):
    def __init__(self):
        self.batch_size = 128
        self.image_path = '/Users/zsheikhb/Documents/Teaching/Programming/pytorch-vae/cropped_images_training_rev1/'
        self.z_dim = 512  #
        self.lr = 2*1e-6
        self.EPOCH = 25
        self.img_scale = 64
        self.use_gpu = torch.cuda.is_available()

config = Config()

train_data_loader = get_dataset(config.image_path, config.img_scale, config.batch_size)


sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Let's use {}".format(device))
model = VAE()
model = model.to(device)


# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# Start training
for epoch in range(config.EPOCH):
    for i, data in enumerate(train_data_loader):
        # Forward pass
        x = data
        x = x.to(device)
        # x_reconst, mu, log_var = model(x)
        x_reconst, z_means, z_log_var = model(x)

        # reconstruction loss
        #reconst_loss = F.mse_loss(x_reconst, x, reduction="sum")
        reconst_loss = F.binary_cross_entropy_with_logits(x_reconst, x)

        # kl divergence
        kl_div = - 0.5 * torch.sum(1 + z_log_var - z_means.pow(2) - z_log_var.exp())

        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, config.EPOCH
                          , i + 1, len(train_data_loader), reconst_loss.item(), kl_div.item()))


    with torch.no_grad():
        z = torch.randn(config.batch_size, config.z_dim, device=device)
        out = model.decoder(z)
        save_image(out, os.path.join(sample_dir, 'sampled-ResNet-{}.png'.format(epoch+1)))
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 3, 64, 64), out.view(-1, 3, 64, 64)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-ResNet-{}.png'.format(epoch+1)))
