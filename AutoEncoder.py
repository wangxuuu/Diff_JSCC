import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from itertools import chain
from models.autoencoder import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
import os
import argparse

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder()
    print_model(autoencoder.encoder, autoencoder.decoder)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--sample_dir", type=str, default='./results/cifar_test/')
    parser.add_argument("--dataset_dir", type=str, default='../data/cifar10/')
    parser.add_argument("--ckpt_dir", type=str, default='./checkpoints/')
    args = parser.parse_args()

    # Create model
    autoencoder = create_model()

    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load(args.ckpt_dir))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = Variable(images.cuda())

        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

        exit(0)

    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    # Define an optimizer and criterion
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(args.epochs):
        running_loss = 0.0

        for i, (inputs, y) in enumerate(trainloader, 0):
            inputs = get_torch_vars(inputs)
            y = get_torch_vars(y)
            # ============ Forward ============
            encoded = autoencoder.encode(inputs)
            outputs = autoencoder.decode(encoded)
            loss = criterion(outputs, inputs)

            # ============ Backward ============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # ============ Logging ============
            running_loss += loss.data
            if (i+1) % 200 == 0:
                print('Epoch [%d, %5d] Step [%d, %5d] loss: %.3f ' %
                      (epoch + 1, args.epochs, i+1, len(trainloader), running_loss / 200))
                running_loss = 0.0
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                # Save the reconstructed images
                x_concat = torch.cat([inputs, outputs], dim=3)
                save_image(x_concat, os.path.join(args.sample_dir, 'reconst-{}.png'.format(epoch+1)), nrow=4)


    print('Finished Training')
    print('Saving Model...')

    torch.save(autoencoder.state_dict(), os.path.join(args.ckpt_dir+'autoencoder.pt'))


if __name__ == '__main__':
    main()