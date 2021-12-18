from sys import stderr
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os

import os, sys
import auxModels
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utilities.attack import ifgsm
from utilities.fftaugment import highPassFilter, highPassNoise, lowPassFilter
from utilities.utils import progress_bar
from reverseResnet import *

model_root = '/tmp2/aislab/OrdinaryHuman/checkpoint/'
num_workers = 8
best_acc = 0
start_epoch = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = torchvision.datasets.CIFAR100
data_root = '/tmp2/dataset/'
data_setfolder = 'cifar100/'
batch_size = 64
# image_shape = (32)
#======================
transform_test = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

# testset = data(root=data_root + data_setfolder, split='val', transform=transforms.Compose(transform_test))
testset = data(root=data_root + data_setfolder, train=False, transform=transforms.Compose(transform_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
#=====================================================
criterion = nn.CrossEntropyLoss()
n_step = int(sys.argv[1])
std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
epsilon = 8 / 255/ std
alpha = epsilon/ n_step

#====================================================
model_names = sys.argv[2:]
for model_name in model_names:
    assert os.path.isdir(model_root), 'Error: no checkpoint directory found!'
    # net = auxModels.resnet18(nAux = 7).to(device)
    # net = auxModels.reverseAuxModel(nAux=7)
    net = ResNet18(nAux=7).to(device)
    try:
        checkpoint = torch.load(f'{model_root}{model_name}.pth')
        net.load_state_dict(checkpoint['net'])
    except:
        net = torch.nn.DataParallel(net, device_ids=[0])
        checkpoint = torch.load(f'{model_root}{model_name}.pth')
        net.load_state_dict(checkpoint['net']) 
    net.eval()
    print(f'Model {model_name}', file=stderr)
# ========================Model========================
    if True:
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
        test_loss /= len(testloader)
        test_acc = 100. * correct/total
        print(f'Testing acc: {test_acc:.5f}, loss: {test_loss:.5f}', file=stderr)
# ===================ATTACK==================================
    if True:
        correct = 0
        total = 0
        attack_loss = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = ifgsm(model=net,x = inputs,y = targets,loss_fn=criterion,epsilon=epsilon,alpha=alpha,num_iter=n_step)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                attack_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (attack_loss/(batch_idx+1), 100.*correct/total, correct, total))
        attack_loss /= len(testloader)
        attack_acc = 100. * correct/total
        print(f'Attacked with {n_step} step(s) acc: {attack_acc:.5f}, loss: {attack_loss:.5f}', file=stderr)
    print(f'=========================================================================', file=stderr)
