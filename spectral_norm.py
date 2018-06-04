import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
import torchvision
import torchvision.transforms as transforms
import tqdm

parser = argparse.ArgumentParser(description='Computing spectral norm of Hessian')

# Directories
parser.add_argument('--data', type=str, default='/u/arpitdev/data/',
                    help='location of the data corpus')
parser.add_argument('--save_dir', type=str, default='default/',
                    help='dir path (inside root_dir) to save the log and the final model')

# Hyperparams
parser.add_argument('--epoch', type=int, default=1,
                    help='compute hessian for this epoch')
parser.add_argument('--M', type=int, default=15,
                    help='number of power iterations (default 15)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# Meta arguments: Tracking, resumability, CUDA
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name (cifar10)')
parser.add_argument('--datasize', type=int, default=45000,
                    help='dataset size')

parser.add_argument('--arch', type=str, default='vgg11',
                    help='arch name (resnet, vgg11)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--cluster', action='store_true', help='do not show the progress bar for batch job')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
### Set the random seed
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset_hes = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)
    train_sampler_hes = torch.utils.data.sampler.SubsetRandomSampler(range(25000))
    trainloader_hessian = torch.utils.data.DataLoader(trainset_hes, batch_size=500, shuffle=False,#shuffle=True, num_workers=2)
                                              sampler=train_sampler_hes, num_workers=2)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    nb_classes = len(classes)
elif args.dataset=='mnist':

        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

        train_set_hes = torchvision.datasets.MNIST(root=args.data, train=True, transform=trans, download=True)
        train_sampler_hes = torch.utils.data.sampler.SubsetRandomSampler(range(45000))
        trainloader_hessian = torch.utils.data.DataLoader(train_set_hes, batch_size=500,  #shuffle=True, num_workers=2)
                                              sampler=train_sampler_hes, num_workers=2)


        nb_classes = 10


print('==> Loading model {}/epoch_{}.batch_0.pt'.format(args.save_dir, args.epoch))
assert os.path.isdir(args.save_dir), 'Error: no checkpoint directory found!'
with open(args.save_dir + '/epoch_{}.batch_0.pt'.format(args.epoch), 'rb') as f:
    checkpoint = torch.load(f)
model = checkpoint['net']

criterion = nn.CrossEntropyLoss()

def hessian_spectral_norm_approx(model, loader, M=args.M, seed=777):
    model.train()
    def get_Hv(v):
        v.volatile=False
        # Calculate gradient
        flat_grad_loss = None
        flat_Hv = None
        ind = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            ind+=1
            model.zero_grad()

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_grad_loss = torch.cat([grad.view(-1) for grad in grads])


            grad_dot_v = (flat_grad_loss * (v)).sum()

            Hv = torch.autograd.grad(grad_dot_v, model.parameters())
            if flat_Hv is None:
                flat_Hv = torch.cat([grad.contiguous().view(-1) for grad in Hv])
            else:
                flat_Hv.data.add_(torch.cat([grad.contiguous().view(-1) for grad in Hv]).data)

        flat_Hv.data.mul_(1./ind)

        return flat_Hv

    p_order = [p[0] for p in model.named_parameters()]
    params = model.state_dict()
    init_w = np.concatenate([params[w].cpu().numpy().reshape(-1,) for w in p_order])

    rng = np.random.RandomState(seed)
    v = Variable(torch.from_numpy(rng.normal(0.0, scale=1.0, size=init_w.shape).astype("float32")).cuda())
    spec_norm_list = []
    for i in tqdm.tqdm(range(M), total=M):

        Hv = get_Hv(v)
        pmax = torch.max(Hv.data)
        nmax = torch.min(Hv.data)
        if pmax<np.abs(nmax):
            spec_norm = nmax
        else:
            spec_norm = pmax
        v=Hv
        v.data.mul_(1./spec_norm)

        print('iter: {}, estimated spectral norm: {} '.format(i, spec_norm))
        spec_norm_list.append(spec_norm)
    return spec_norm_list


spec_norm_list = hessian_spectral_norm_approx(model, trainloader_hessian)
with open(args.save_dir + '/model_for_hessian_epoch_' + str(args.epoch) + '.pkl', "wb") as f:
        pkl.dump(spec_norm_list[-1], f)
print('Spectral norm computed for ', str(args.epoch), ' saved at: ', args.save_dir + '/model_for_hessian_epoch' + str(args.epoch) + '.pkl')
print('Estimated spectral norm: ', spec_norm_list[-1])
