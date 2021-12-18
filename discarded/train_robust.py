import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import argparse

from models import *
from utils import *
from fftaugument import *
from robustdata import *

parser = argparse.ArgumentParser(
    description='PyTorch model trained with FFT filter')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--nepoch', default=200, type=int, help='number of epoch')
parser.add_argument('--low', default=0.25, type=float,
                    help='Critical value of low pass fft')
parser.add_argument('--high', default=0.25, type=float,
                    help='Critical value of high noise fft')
parser.add_argument('--eps', default=0.1, type=float,
                    help='Ratio of high frequency and img')
parser.add_argument('--name', default='ckpt', type=str, help='ckptName',)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
# Data
print('==> Preparing data..')

data = torchvision.datasets.CIFAR10
data_root = '/tmp2/b08902011/Data/cifar10/'
batch_size = 512
num_workers = 8
robust_cifar_data, robust_cifar_label = fetch_data(
    "d_robust_CIFAR", "/tmp2/b08902011/Data/robustdata/")
# nonrobust_cifar_data, nonrobust_cifar_label = fetch_data(
#     "d_non_robust_CIFAR", "/tmp2/b08902011/Data/robustdata/")

transform_train = transforms.Compose([
    transforms.RandomAffine(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(1.2, p=0.5),
    transforms.RandomInvert(p=0.5),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = RobustDataset(robust_cifar_data, robust_cifar_label, transform_train)
trainloader = torch.utils.data.DataLoader(trainset
    , batch_size=batch_size,    shuffle=True,    num_workers=num_workers,)

testset = data(root=data_root, train=False,
                download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# ========================Model========================
print('==> Building model..')
net = ResNet18().to(device)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.name}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(
#     net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.nepoch)


def train(epoch):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    Acc = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), acc, correct, total))
        Acc += acc
    return Acc / len(trainloader)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    acc = 100. * correct/total
    if acc > best_acc:
        best_acc = acc
        save_name = f'./checkpoint/{args.name}_best.pth'
        print('Saving best..')
    else:
        save_name = f'./checkpoint/{args.name}.pth'
        print('Saving..')

    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }

    torch.save(state, save_name)
    return acc


training = []
testing = []
for epoch in range(start_epoch, start_epoch+args.nepoch):
    training.append(train(epoch))
    testing.append(test(epoch))
    scheduler.step()
    save_proc(training=training, testing=testing,
              name=args.name, folder="curve")

# net = VGG('VGG19')
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
