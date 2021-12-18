import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import auxModels
import PIL
import os
import sys
if "MyLib":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
    from utils import *
    from utilities.fftaugment import *
    from utilities.utils import SRNG, progress_bar
model_root = '/tmp2/aislab/OrdinaryHuman/checkpoint/'
data_root = '/tmp2/dataset/cifar100'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Auxiliary Learning')
parser.add_argument('--epoch', default=210, type=int)
parser.add_argument('--name', default="ckpt", type=str)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.05, type=float)
parser.add_argument('--gama', default=5.0, type=float)
parser.add_argument('--rotate', default=20, type=float)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--noise', default=10, type=float)
parser.add_argument('--blur', default=0.5, type=float)
parser.add_argument('--width', default=64, type=int)
parser.add_argument('--hfc', action="store_true")
parser.add_argument('--low', action="store_true")
parser.add_argument('--smooth', action="store_true")
args = parser.parse_args()
BATCH_SIZE = 128
LR = 0.1

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


trainset = torchvision.datasets.CIFAR100(
    root=data_root,
    train=True,
    download=False,
    transform=transform_train
)

testset = torchvision.datasets.CIFAR100(
    root=data_root,
    train=False,
    download=False,
    transform=transform_test
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

trans_hfcnoise = transforms.Compose([
    highPassNoise(0.2, 0.5, trainset, srng=SRNG(0xC8763, len(trainset))),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trans_lowpass = transforms.Compose([
    lowPassFilter(0.2),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trans = [trans_clean, trans_blur,
           trans_noise, trans_contrast, trans_rotate]
if args.hfc:
    trans.append(trans_hfcnoise)
if args.low:
    trans.append(trans_lowpass)


if args.depth == 18:
    net = auxModels.resnet18(nAux=len(trans))
elif args.depth == 50:
    net = auxModels.resnet50(nAux=len(trans))
elif args.depth == 101:
    net = auxModels.resnet101(nAux=len(trans))
elif args.depth == 152:
    net = auxModels.resnet152(nAux=len(trans))
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

best_acc = 0
smooth = torch.tensor(1e-4,device=device)
for epoch in range(args.epoch):
    if epoch in [60, 140, 180]:
        smooth *= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    weight_loss_all = 0.0
    print(f'Epoch: {epoch + 1}')
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs, labels.cuda()

        auxiliary_inputs = get_auxiliary_data(inputs, trans)

        outputs = net(auxiliary_inputs)
        
        output_main = outputs[0]
        loss = criterion(output_main, labels)
        output_main = output_main.detach()

        for output in outputs[1:]:
            loss += criterion(output, labels) * args.alpha
            loss += CrossEntropy(output, output_main) * args.beta

        main_w, main_b = net.auxiliary_classifiers_list[0].fc.parameters()
        weight_loss =torch.empty(size=([]),dtype=torch.float32,device=device)
        for aux_net in net.auxiliary_classifiers_list[1:]:
            fc_w, fc_b = aux_net.fc.parameters()
            weight_loss += ((main_w - fc_w) ** 2).sum() + \
                ((main_b - fc_b) ** 2).sum()
        if args.smooth:
            loss += weight_loss * smooth
        elif epoch >= 180:
            loss += weight_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        weight_loss_all += weight_loss.item()
        _, predicted = torch.max(output_main, 1)
        total += float(labels.size(0))
        correct += float(predicted.eq(labels).cpu().sum())
        progress_bar(i, len(trainloader), 'Loss: %.03f Weight Loss: %0.3f | Acc: %.4f%%' % (
            sum_loss / (i + 1), weight_loss_all / (i+1), 100 * correct / total))

    save_name = f'{model_root}{args.name}.pth'
    print('Saving..')
    state = {
        'net': net.state_dict(),
    }
    torch.save(state, save_name)
    
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        net.eval()
        for image, labels in testloader:
            image, labels = image.to(device), labels.to(device)
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            total += float(labels.size(0))
            correct += float((predicted == labels).sum())
        acc1 = (100 * correct/total)
    print('Test Set Accuracy: %.4f%%' % (acc1))
