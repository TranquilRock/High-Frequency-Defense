from sys import stderr
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import os

from models import *
from utilities.attack import ifgsm
from utilities.fftaugment import highPassFilter, highPassNoise, lowPassFilter
from utilities.utils import progress_bar
from utilities.environment import device, args, data_root, model_root
from utilities.environment import *
        

num_workers = 8
best_acc = 0
start_epoch = 0
print('==> Preparing data..')
data = torchvision.datasets.ImageNet
data_setfolder = 'imagenet2012/'
batch_size = 64
image_shape = (224,224)
#======================
transform_test = [
    transforms.Resize(image_shape), 
    transforms.ToTensor(),
]

if args.low >= 0:
    transform_test.append(lowPassFilter(args.low))
if args.high >= 0:
    transform_test.append(highPassFilter(args.high))
if args.noise >= 0:
    transform_noise = transforms.Compose([
                                transforms.Resize(image_shape), 
                                transforms.ToTensor()])
    noiseSet = data(root=data_root + data_setfolder, split='test', transform=transform_noise)
    transform_test.append(highPassNoise(args.noise, args.eps, noiseSet))
transform_test.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

testset = data(root=data_root + data_setfolder, split='val', transform=transforms.Compose(transform_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
#=====================================================
criterion = nn.CrossEntropyLoss()
n_step = args.nepoch
std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(3, 1, 1)
epsilon = 8 / 255/ std
alpha = epsilon/ n_step

#====================================================
model_names = ['tune_noise25','tune_dwl_low25_noise25','tune_dwl_noise25','tune_low25noise25',]
for model_name in model_names:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(model_root), 'Error: no checkpoint directory found!'
    if model_name == 'tune_dwl_low25_noise25' or model_name == 'tune_low25noise25' or model_name == 'tune_noise25':
        net = torchvision.models.resnet50(False, True,).to(device)
        net = torch.nn.DataParallel(net, device_ids=[0])
    else:
        net = torchvision.models.resnet50(False, True,).to(device)
    checkpoint = torch.load(f'{model_root}{model_name}.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    best_acc = checkpoint['acc']
    print(f'Model {model_name} with best_acc: {best_acc}', file=stderr)
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
