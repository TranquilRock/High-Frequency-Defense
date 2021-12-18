import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from models import *

from utilities.fftaugment import highPassFilter, highPassNoise, highPassNoiseTarget, lowPassFilter
from utilities.lossFunction import dynamicWeightedCELoss
from utilities.utils import progress_bar, save_proc, SRNG
from utilities.environment import device, args, data_root, model_root, meta_root
from utilities.environment import *

best_acc = 0
start_epoch = 0
print('==> Preparing data..')
data = torchvision.datasets.ImageNet
data_setfolder = 'imagenet2012/'
batch_size = args.batchsize
image_shape = (224,224)
transform_train = [
    transforms.RandomAffine(10),
    transforms.RandomResizedCrop(image_shape),
    transforms.ColorJitter(.4,.4,.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
transform_trainlabel = []

if args.low >= 0:
    transform_train.append(lowPassFilter(args.low))
if args.high >= 0:
    transform_train.append(highPassFilter(args.high))
if args.noise >= 0:
    transform_noise = transforms.Compose([
    transforms.Resize(image_shape), transforms.ToTensor()])
    noiseSet = data(root=data_root + data_setfolder, split='train', transform=transform_noise)
    transform_train.append(highPassNoise(args.noise, args.eps, noiseSet,srng=SRNG(0xC8763,len(noiseSet))))
    if args.dynamicWeightLoss:
        transform_trainlabel.append(highPassNoiseTarget(noiseSet,srng=SRNG(0xC8763,len(noiseSet))))
    
transform_train.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
transform_test = [
    transforms.Resize(image_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
]

trainset = data(root=data_root + data_setfolder, split='train', transform=transforms.Compose(transform_train),target_transform=transforms.Compose(transform_trainlabel))
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = data(root=data_root + data_setfolder, split='val', transform=transforms.Compose(transform_test))
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=8)

# ========================Model========================
print('==> Building model..')

# net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101',
#                      pretrained=True).to(device)
# net = torchvision.models.efficientnet_b2(True, True,).to(device)

net = torchvision.models.resnet50(True, True,).to(device)
if device == 'cuda':
    if args.dataParallel > 0:
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(args.dataParallel)])
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(model_root), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'{model_root}{args.name}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.dynamicWeightLoss and args.noise >= 0:
    criterion = dynamicWeightedCELoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepoch)


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
        if args.dynamicWeightLoss and args.noise >= 0:
            correct += predicted.eq(targets[:,0]).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), acc, correct, total))
        Acc += acc
    return Acc / len(trainloader)


def test(epoch):
    global best_acc
    criterion = nn.CrossEntropyLoss()
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
                         % (test_loss/(batch_idx + 1), 100.*correct/total, correct, total))

#====================== Save checkpoint.============================================
    if not os.path.isdir(model_root):
        os.mkdir(model_root)
        
    acc = 100. * correct/total
    if acc > best_acc:
        best_acc = acc
        save_name = f'{model_root}{args.name}_best.pth'
        print('Saving best..')
    else:
        save_name = f'{model_root}{args.name}.pth'
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

for resume_routine in range(start_epoch):
    scheduler.step()
for epoch in range(start_epoch, start_epoch+args.nepoch):
    training.append(train(epoch))
    testing.append(test(epoch))
    scheduler.step()
    save_proc(training = training, testing = testing, name = args.name,folder = meta_root)