__author__ = 'Chen Xing, Devansh Arpit'
import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
from models import ResNet56, vgg11, MLPNet
from utils import progress_bar
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Experiments of "A Walk with SGD"')

# Directories
parser.add_argument('--data', type=str, default='default/data/',
                    help='location of the data corpus')
parser.add_argument('--save_dir', type=str, default='default/',
                    help='dir path (inside root_dir) to save the log and the final model')

# Hyperparams
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate factor that gets multiplied to the hard coded LR schedule in the code')
parser.add_argument('--epochs', type=int, default=4000,
                    help='upper epoch limit')
parser.add_argument('--init', type=str, default="he")
parser.add_argument('--wdecay', type=float, default=1e-4,
                    help='weight decay applied to all weights')
parser.add_argument('--bs', type=int, default=45000, metavar='N',
                    help='batch size')
parser.add_argument('--mbs', type=int, default=5000, metavar='N',
                    help='minibatch size')
parser.add_argument('--bn', type=bool, default=True,
                    help='batch norm T/F')

# Meta arguments: Tracking, resumability, CUDA
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name (cifar10)')

parser.add_argument('--datasize', type=int, default=45000,
                    help='dataset size')

parser.add_argument('--arch', type=str, default='resnet',
                    help='arch name (resnet, vgg11)')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume experiment ')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--cluster', action='store_true', help='do not show the progress bar for batch job')
parser.add_argument('--noise_var', type=float, default=0.,
                    help='')
parser.add_argument('--opt', type=str, default='sgd',
                    help='')
parser.add_argument('--save_model_per_iter', type=bool, default=True,
                    help='')
args = parser.parse_args()


def get_correct_lr(lr):
    global args
    assert args.bs >= args.mbs
    assert args.bs % args.mbs == 0  ## batch size must be a multiple of effective mini batch size

    ### correction for gradient summing
    grad_correct_factor = args.mbs / float(args.bs)
    correct_lr = lr * grad_correct_factor
    lr = correct_lr
    return lr

if args.arch == 'resnet':
    lr_sch = [[1000000000000, 1]]
    mom_sch = [[99999999, 1.]]
elif args.arch == 'vgg11':
    lr_sch = [[1000000000000, 1]]
    mom_sch = [[99999999, 1.]]
elif args.arch == 'mlp':
    lr_sch = [[1000000000000, 1]]
    mom_sch = [[99999999, 1.]]

use_cuda = torch.cuda.is_available()
### Set the random seed
torch.manual_seed(args.seed)
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
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(args.datasize))
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(45000, 50000))
    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs,
                                              sampler=train_sampler, num_workers=2)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs,
                                              sampler=valid_sampler, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    nb_classes = len(classes)
elif args.dataset == 'mnist':

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = torchvision.datasets.MNIST(root=args.data, train=True, transform=trans, download=True)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(args.datasize))
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000, 60000))

    test_set = torchvision.datasets.MNIST(root=args.data, train=False, transform=trans)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs,  # shuffle=True, num_workers=2)
                                              sampler=train_sampler, num_workers=2)
    validloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs,  # shuffle=True, num_workers=2)
                                              sampler=valid_sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mbs, shuffle=False, num_workers=2)

    nb_classes = 10

elif args.dataset == 'imagnet':
    # Data loading code
    transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    train = datasets.ImageFolder(traindir, transform)
    val = datasets.ImageFolder(valdir, transform)
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=args.mbs, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(
        val, batch_size=args.mbs, shuffle=True, num_workers=2)
###############################################################################
# Build the model
###############################################################################
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.save_dir), 'Error: no checkpoint directory found!'
    with open(args.save_dir + '/best_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
    model = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    lr_list = pkl.load(open(args.save_dir + "/LR_list.pkl", "r"))

    train_loss_list = pkl.load(open(args.save_dir + "/train_loss.pkl", "r"))

    train_acc_list = pkl.load(open(args.save_dir + "/train_acc.pkl", "r"))

    valid_acc_list = pkl.load(open(args.save_dir + "/valid_acc.pkl", "r"))

    param_dist_list = pkl.load(open(args.save_dir + "/param_dist_list.pkl", "r"))

else:
    if os.path.isdir(args.save_dir):
        with open(args.save_dir + '/log.txt', 'w') as f:
            f.write('')
    print('==> Building model..')
    start_epoch = 1
    if args.arch == 'resnet':
        model = ResNet56(dropout=0, bn=args.bn)
    elif args.arch == 'vgg11':
        model = vgg11(dropout=0, bn=args.bn)
    elif args.arch == 'mlp':
        model = MLPNet(dropout=0, bn=args.bn)
    nb = 0
    if args.init == 'he':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nb += 1
                print('Update init of ', m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                print('Update init of ', m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    best_acc = 0
    lr_list = []

    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    param_dist_list = []
    cos_mg_list = []
    grad_norm_list = []

if args.cuda:
    model.cuda()
total_params = sum(np.prod(x.size()) if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
with open(args.save_dir + '/log.txt', 'a') as f:
    f.write(str(args) + ',total_params=' + str(total_params) + '\n')

criterion = nn.CrossEntropyLoss()
criterion_nll = nn.NLLLoss()

print('Saving initial model..')
state = {
    'net': model,
    'acc': -1,
    'epoch': 0,
}
with open(args.save_dir + '/init_model.pt', 'wb') as f:
    torch.save(state, f)

###############################################################################
# Training code
###############################################################################
def compute_dist():
    global model
    global param_dist_list
    with open(args.save_dir + '/init_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
    model_init = checkpoint['net']
    d = 0.
    for param1, param2 in zip(model_init.parameters(), model.parameters()):
        param1 = param1.data.cpu().numpy()
        param2 = param2.data.cpu().numpy()
        d += np.sum((param1 - param2) ** 2)
    param_dist_list.append(np.sqrt(d))

    with open(args.save_dir + "/param_dist_list.pkl", "wb") as f:
        pkl.dump(param_dist_list, f)

def test(epoch, loader, valid=False):
    global best_acc
    global model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if not args.cluster:
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if valid and acc > best_acc:
        print('Saving best model..')
        state = {
            'net': model,
            'acc': acc,
            'epoch': epoch,
        }
        with open(args.save_dir + '/best_model.pt', 'wb') as f:
            torch.save(state, f)
        best_acc = acc
    return acc


def compute_angle_et_norm():
    global optimizer
    global model
    global cos_mg_list
    global grad_norm_list
    if not hasattr(compute_angle_et_norm, 'm'):
        compute_angle_et_norm.m = {}

    dot = 0
    norm1 = 0
    norm2 = 0
    for name, variable in model.named_parameters():

        g = variable.grad.data
        if not name in compute_angle_et_norm.m.keys():
            compute_angle_et_norm.m[name] = g.clone()
            dot += torch.sum(compute_angle_et_norm.m[name] * g.clone())
            norm1 += torch.sum(compute_angle_et_norm.m[name] * compute_angle_et_norm.m[name])
            norm2 += torch.sum(g.clone() * g.clone())
        else:
            dot += torch.sum(compute_angle_et_norm.m[name] * g.clone())
            norm1 += torch.sum(compute_angle_et_norm.m[name] * compute_angle_et_norm.m[name])
            norm2 += torch.sum(g.clone() * g.clone())
            compute_angle_et_norm.m[name] = g.clone()

    cos = dot / (np.sqrt(norm1 * norm2) + 0.000001)
    cos_mg_list.append(cos)
    with open(args.save_dir + "/cos_mg_list.pkl", "wb") as f:
        pkl.dump(cos_mg_list, f)

    grad_norm_list.append(norm2)
    with open(args.save_dir + "/grad_norm_list.pkl", "wb") as f:
        pkl.dump(grad_norm_list, f)

    return cos

def train(epoch):
    global trainloader
    global optimizer
    global args
    global model
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_list = []
    for lr_ in lr_sch:
        if epoch <= lr_[0]:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * lr_[1],
                                        momentum=0,
                                        weight_decay=args.wdecay)
            break

    optimizer.zero_grad()
    if not hasattr(train, 'nb_samples_seen'):
        train.nb_samples_seen = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if train.nb_samples_seen + args.mbs== args.bs:
            optimizer.step()
            compute_angle_et_norm()
            compute_dist()
            optimizer.zero_grad()
            if args.save_model_per_iter is True and epoch<=450:
                print('Saving intermediate model..')
                state = {
                    'net': model,
                    'iter': epoch,
                }
                with open(args.save_dir + '/epoch_{}.pt'.format(epoch), 'wb') as f:
                    torch.save(state, f)
            train.nb_samples_seen = 0
        else:
            train.nb_samples_seen += args.mbs
        loss_list.append(loss.data[0])
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if not args.cluster:
            progress_bar(batch_idx, len(trainloader), 'Epoch {:3d} | Loss: {:3f} | Acc: {:3f}'
                         .format(epoch, train_loss / (batch_idx + 1), 100. * correct / total))

    return sum(loss_list) / float(len(loss_list)), 100. * correct / total
# Loop over epochs.

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                            momentum=0, \
                                            weight_decay=args.wdecay)
for epoch in range(start_epoch, args.epochs + 1):
    epoch_start_time = time.time()
    loss, train_acc = train(epoch)
    print("current lr:"+str(optimizer.param_groups[0]['lr']))
    train_loss_list.append(loss)
    train_acc_list.append(train_acc)
    valid_acc = test(epoch, validloader, valid=True)
    valid_acc_list.append(valid_acc)
    with open(args.save_dir + "/train_loss.pkl", "wb") as f:
        pkl.dump(train_loss_list, f)
    with open(args.save_dir + "/train_acc.pkl", "wb") as f:
        pkl.dump(train_acc_list, f)
    with open(args.save_dir + "/valid_acc.pkl", "wb") as f:
        pkl.dump(valid_acc_list, f)
    lr_list.append(optimizer.param_groups[0]['lr'] / (args.mbs / float(args.bs)))
    epoch_fac = 1.
    status = 'Epoch {}/{} | Loss {:3f} | Acc {:3f} | val-acc {:.3f}| max-variance {:.3f}| LR {:4f} | BS {}'. \
        format(epoch, args.epochs * epoch_fac, loss, train_acc, valid_acc, 0, lr_list[-1], args.bs)
    with open(args.save_dir + '/log.txt', 'a') as f:
        f.write(status + '\n')

    with open(args.save_dir + "/LR_list.pkl", "wb") as f:
        pkl.dump(lr_list, f)
    print('-' * 89)

# Load the best saved model.
with open(args.save_dir + '/best_model.pt', 'rb') as f:
    best_state = torch.load(f)
model = best_state['net']
# Run on test data.
test_acc = test(epoch, testloader, valid=True)
print('=' * 89)
print('| End of training | test acc {}'.format(test_acc))
print('=' * 89)

