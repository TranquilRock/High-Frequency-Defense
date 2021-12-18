from torch import tensor
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import os
import sys
import time
import torch.nn as nn
import torch.nn.init as init

import torch


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def save_proc(training, testing, name='process', folder='.'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    plt.plot([i for i in range(1, len(training) + 1)], training, label='train')
    plt.plot([i for i in range(1, len(testing) + 1)], testing, label='test')
    plt.legend()
    plt.savefig(f'{folder}/{name}.png')
    plt.clf()


def saveTensorImg(source: tensor, folder: str, name: str):
    save_image(source, f'{folder}/{name}.png')  # .permute(2, 0, 1)


def print_data(data_set, figure=1, s=[]):
    datas = [data_set[i] for i in range(s)]
    for i, pic in enumerate(datas):
        if(i < s):
            continue
        if(i > s + 3):
            break
        plt.figure(figure)
        plt.subplot(2, 2, i - s + 1)
        plt.imshow(pic.permute(1, 2, 0))




class SRNG():
    def __init__(self, seed, N) -> None:
        self.seed = seed
        self.N = N

    def gen(self):
        x = self.seed
        self.seed = (self.seed * 11) % self.N
        return ((x ** 3) - 7 * (x ** 2) + 2 * x + 5) % self.N

def get_methods(obj, spacing=20):
    methodList = []
    for method_name in dir(obj):
        try:
            if callable(getattr(obj, method_name)):
                methodList.append(str(method_name))
        except:
            methodList.append(str(method_name))
    processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(str(method.ljust(spacing)) + ' ' +
                processFunc(str(getattr(obj, method).__doc__)[0:90]))
        except:
            print(method.ljust(spacing) + ' ' + ' getattr() failed')
    print("===================")
    print(obj)
    print("===================")
    vars(obj)