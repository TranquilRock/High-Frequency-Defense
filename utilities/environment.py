from sys import stderr
import os
import torch
import torch.backends.cudnn as cudnn
import argparse
parser = argparse.ArgumentParser(
    description='PyTorch model trained with FFT filter')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--nepoch', default=90, type=int, help='number of epoch')

parser.add_argument('--low', default = -1, type=float,
                    help='Critical value of low pass fft.')
parser.add_argument('--high', default = -1, type=float,
                    help='Critical value of high noise fft.')
parser.add_argument('--noise', default = -1, type=float,
                    help='Critical value of high noise fft.')
parser.add_argument('--eps', default = 0.1, type=float,
                    help='Ratio of mixing high frequency and img')
parser.add_argument('--batchsize', default = 128, type=int,
                    help='Training batchsize.')
parser.add_argument('--mixLabel', action='store_true',
                    help='Add noise label to target.')
parser.add_argument('--resume', action='store_true',
                    help='Resume from checkpoint.')
parser.add_argument('--dynamicWeightLoss', action='store_true',
                    help='Use custom loss function.')
parser.add_argument('--dataParallel', default = 1, type=int,
                    help='Apply torch\'s dataparallel with n GPU.')
parser.add_argument('--name', default='ckpt', type=str, help='Name of check point.',)

args = parser.parse_args()
print(args)
# print(f"Model {args.name} with Args: Lowpass:{args.low} Highpass:{args.high} \
#         Highnoise:{args.noise} eps:{args.eps} \
#         {'with custom loss' if args.dynamicWeightLoss else ''}\
#         {'with custom loss' if args.dynamicWeightLoss else ''}\
#         {'with custom loss' if args.dynamicWeightLoss else ''}"\
#         ,file=stderr)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True
if os.path.isdir('/home/OrdinaryHuman'):
    data_root = '/tmp2/dataset/'
    model_root = '/tmp2/aislab/OrdinaryHuman/checkpoint/'
    meta_root = '/tmp2/aislab/OrdinaryHuman/metadata/'
else:
    data_root = '/tmp2/dataset/'
    model_root = '/tmp2/b08902011/checkpoint/'
    meta_root = '/tmp2/b08902011/metadata/'
