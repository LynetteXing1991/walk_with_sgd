__author__ = 'Chen Xing, Devansh Arpit'
import numpy as np
from models import ResNet56, vgg11, MLPNet
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl
import torchvision
import torchvision.transforms as transforms
import argparse
import torchvision.datasets as datasets
parser = argparse.ArgumentParser(description='Experiments of "A Walk with SGD"')

# Directories
parser.add_argument('--data', type=str, default='/default/data',
                    help='location of the data corpus')
parser.add_argument('--model_dir', type=str, default='',
                    help='')
parser.add_argument('--save_dir', type=str, default='',
                    help='')
# Hyperparams
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--bs', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--mbs', type=int, default=100, metavar='N',
                    help='minibatch size')
# Meta arguments: Tracking, resumability, CUDA
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset name (cifar10)')
parser.add_argument('--datasize', type=int, default=45000,
                    help='dataset size')
parser.add_argument('--arch', type=str, default='resnet',
                    help='arch name (resnet, vgg11)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--epoch_index', type=str, default='1',
                    help='resume experiment ')
parser.add_argument('--num_batches', type=int, default=450,
                    help='number of batches per epoch')
parser.add_argument('--mode', type=str, default='sgd',
                    help='mode name (sgd, gd)')




args = parser.parse_args()

criterion = nn.CrossEntropyLoss()
criterion_nll = nn.NLLLoss()


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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs,  # shuffle=True, num_workers=2)
                                              sampler=train_sampler, num_workers=2)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.mbs,  # shuffle=True, num_workers=2)
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
    trainloader_variance = torch.utils.data.DataLoader(train_set, batch_size=50,  # shuffle=True, num_workers=2)
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

def test(epoch, model, loader):
    model.train()
    test_loss = 0
    correct = 0
    total = 0
    optimizer=torch.optim.SGD(model.parameters(),lr=0)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    test_loss/=batch_idx
    # Save checkpoint.
    acc = 100. * correct / total
    print("Loss of "+str(epoch)+': '+str(test_loss) +"  Accuracy: "+str(acc))
    return acc, test_loss


def iteratively_interpolate_model(dir,save_dir):
    torch.nn.Module.dump_patches = True
    with open(dir + "epoch_" + args.epoch_index + '.batch_0.pt', 'rb') as f:
    #with open(dir + 'init_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
        model_initial = checkpoint['net']
    model1=model_initial
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, dist_iteration_list=[],[],[],[],[]
    e=0
    for j in range(1,args.num_batches):
        with open(dir + "epoch_" + args.epoch_index + ".batch_" + str(j) + '.pt', 'rb') as f:
            checkpoint = torch.load(f)
        model2 = checkpoint['net']
        e = interpolate_between_2models(model1,model2,train_loss_list,train_acc_list, e)
        model1 = model2
        print("Iteration Number: "+str(len(train_loss_list)))
        with open(save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)

        with open(save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)

def iteratively_interpolate_model_gd(dir,save_dir):
    torch.nn.Module.dump_patches = True
    with open(dir + "epoch_1", 'rb') as f:
    #with open(dir + 'init_model.pt', 'rb') as f:
        checkpoint = torch.load(f)
        model_initial = checkpoint['net']
    model1=model_initial
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, dist_iteration_list=[],[],[],[],[]
    e=0
    for j in range(2,args.num_batches):
        with open(dir + "epoch_"  + str(j) + '.pt', 'rb') as f:
            checkpoint = torch.load(f)
        model2 = checkpoint['net']
        e = interpolate_between_2models(model1,model2,train_loss_list,train_acc_list, e)
        model1 = model2
        print("Iteration Number: "+str(len(train_loss_list)))
        with open(save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)

        with open(save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)

def interpolate_between_2models(model1,model2,train_loss_list,train_acc_list,epoch):
    if args.arch == 'resnet':
        model = ResNet56()
    elif args.arch == 'vgg11':
        model = vgg11()
    elif args.arch == 'MLP':
        model = MLPNet()
    model.cuda()
    model1.eval()
    model2.eval()
    model.eval()
    alpha_list = np.arange(0, 1, 0.1)
    for alpha in alpha_list:
        new_dict = {}
        p1_params = model1.state_dict()
        p2_params = model2.state_dict()
        for p in p1_params.keys():
            if p in p2_params.keys():
                new_dict[p] = (1 - alpha) * p1_params[p] + alpha * p2_params[p]
            else:
                print(p)
        model.load_state_dict(new_dict)
        train_acc, train_loss = test(epoch,model, trainloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        epoch+=1
    return epoch
save_dir=args.model_dir+'/interpolation_'+str(args.epoch_index)+'/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.mode=='sgd':
    iteratively_interpolate_model(args.model_dir, args.save_dir)
else:
    iteratively_interpolate_model_gd(args.model_dir, args.save_dir)


