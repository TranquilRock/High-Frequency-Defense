import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import os
import sys
if "MyLib":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
    from utils import *
    from utilities.fftaugment import *
    from utilities.utils import SRNG, progress_bar
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

trans_contrast = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5,
                           saturation=0.5, hue=0),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_noise = transforms.Compose([
    transforms.ToPILImage(),
    GaussianNoise(20),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_blur = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda x: x.filter(PIL.ImageFilter.GaussianBlur(1.0))),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_clean = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_rotate = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


data_root = '/tmp2/dataset/cifar100'
BATCH_SIZE = 128

trainset = torchvision.datasets.CIFAR100(
    root=data_root,
    train=True,
    download=True,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR100(
    root=data_root,
    train=False,
    download=True,
    transform=transform_test
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)

trans_hfcnoise = transforms.Compose([
    highPassNoise(0.2, 0.5, trainset, srng=SRNG(0xC8763, len(trainset))),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans_lowpass = transforms.Compose([
    lowPassFilter(0.2),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
