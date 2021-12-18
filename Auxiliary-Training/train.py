import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import auxModels
import os
import sys
from auxTransforms import *
from reverseResnet import *
if "MyLib":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
    from utils import *
    from utilities.fftaugment import *
    from utilities.utils import progress_bar

model_root = '/tmp2/aislab/OrdinaryHuman/checkpoint/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Auxiliary Learning')
parser.add_argument('--epoch', default=150, type=int)
parser.add_argument('--name', default="ckpt", type=str)

parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.05, type=float)
parser.add_argument('--gama', default=5.0, type=float)
parser.add_argument('--rotate', default=20, type=float)
parser.add_argument('--noise', default=10, type=float)
parser.add_argument('--blur', default=0.5, type=float)
parser.add_argument('--lr', default=0.01, type=float)

parser.add_argument('--hfc', action="store_true")
parser.add_argument('--low', action="store_true")
parser.add_argument('--reverse', action="store_true")

args = parser.parse_args()
print(args)


torch.backends.cudnn.benchmark = True

trans = [trans_clean, trans_blur, trans_noise, trans_contrast, trans_rotate]
if args.hfc:
    trans.append(trans_hfcnoise)
if args.low:
    trans.append(trans_lowpass)


# net = auxModels.reverseAux(nAux=len(trans)).to(device)
net = ResNet18(nAux=len(trans)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)
best_acc = 0
for epoch in range(args.epoch):
    if epoch in [40, 80, 120]:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    sum_weight_loss = 0.0
    sum_aux_loss = 0.0
    print(f'Epoch: {epoch + 1}')
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        auxiliary_inputs = get_auxiliary_data(inputs, trans)

        Prediction = net(auxiliary_inputs[0])
        loss = criterion(Prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

        # ====================================================
        if args.reverse:
            net.linear.requires_grad_ = False
            auxPrediction, *features = net.auxForward(auxiliary_inputs)
            auxLoss = criterion(auxPrediction, labels)

            mainFeature = features[0].detach()
            for feature in features[1:]:
                auxLoss += CrossEntropy(feature, mainFeature) * args.beta

            weight_loss = torch.empty(
                size=([]), dtype=torch.float32, device=device)
            mainExtractor = net.AuxExtractor[0].parameters()
            for extractor in net.AuxExtractor[1:]:
                extractor = extractor.parameters()
                for para1, para2 in zip(extractor, mainExtractor):
                    weight_loss += ((para1 - para2) ** 2).sum()

            if epoch >= 120:
                auxLoss += weight_loss

            optimizer.zero_grad()
            auxLoss.backward()
            optimizer.step()

            net.linear.requires_grad_ = True

            sum_aux_loss += auxLoss.item()
            sum_weight_loss += weight_loss.item()
        # ====================================================
        _, predicted = torch.max(Prediction, 1)
        total += float(labels.size(0))
        correct += float(predicted.eq(labels).cpu().sum())

        # ====================================================
        loss = sum_loss / (i+1)
        weight_loss = sum_weight_loss / (i+1)
        aux_loss = sum_aux_loss / (i+1)
        Acc = 100 * correct / total
        progress_bar(i, len(trainloader),
                     f'Loss: {loss:.3f} | Weight Loss: {weight_loss:.3f} | AuxLoss: {aux_loss:.3f} | Acc: {Acc:.3f}')

    print('Saving..')
    state = {
        'net': net.state_dict(),
    }
    torch.save(state, f'{model_root}{args.name}.pth')

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
