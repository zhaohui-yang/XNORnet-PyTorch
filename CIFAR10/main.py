from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import init
import time
from nin import *
from utils import *

def adjust_lr(optimizer):
  print('Adjust lr')
  for param_group in optimizer.param_groups:
    param_group['lr'] /= 10

def train(args, model, device, train_loader, optimizer, epoch):
  model.train()
  params = torch.zeros(args.batch_size).cuda(); flops = torch.zeros(args.batch_size).cuda()
  for batch_idx, (data, target) in enumerate(train_loader):
    params.zero_(); flops.zero_()
    tic = time.time()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output, params, flops = model(data, params, flops)
    toc = time.time()
    loss = F.cross_entropy(output, target)
    loss.backward()
    
    for layer in model.modules():
      if isinstance(layer, XNORConv2d) or isinstance(layer, XNORLinear):
        layer.copy_grad()
        
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(), (toc-tic)))

def test(args, model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    params = torch.zeros(args.test_batch_size).cuda(); flops = torch.zeros(args.test_batch_size).cuda()
    for data, target in test_loader:
      params.zero_(); flops.zero_()
      data, target = data.to(device), target.to(device)
      output, params, flops = model(data, params, flops)
      test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  return correct / len(test_loader.dataset)

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=320, metavar='N',
            help='number of epochs to train (default: 320)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--gpu', type=list, default=[1],
            help='gpu list')
  parser.add_argument('--log-interval', type=int, default=50, metavar='N',
            help='how many batches to wait before logging training status')
  parser.add_argument('--save-model', action='store_true', default=True,
            help='For Saving the current Model')
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in args.gpu])

  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../../data', train=True, download=True,
             transform=transforms.Compose([
               transforms.ToTensor(),
             ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../../data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
             ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

  model = NIN().to(device)
  print(model)
  #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = 1e-5)
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-5)

  decay_epochs = [120, 200, 240, 280]
  max_acc = 0
  
  countSize(model.state_dict())
  for epoch in range(0, args.epochs):
    if epoch in decay_epochs:
      adjust_lr(optimizer)
    train(args, model, device, train_loader, optimizer, epoch)
    acc = test(args, model, device, test_loader)
    if acc > max_acc:
      max_acc = acc

  params_1, params_32 = countParams(model.state_dict())
  size = countSize(model.state_dict())
    
  if (args.save_model):
    torch.save(model.state_dict(),"cifar10__nin__acc{}__{:.3f}M1bit__{:.3f}M32bit__{:.3f}Mmemory__rnd{}.pth".format(max_acc, params_1 / 1e6, params_32 / 1e6, size / (2**20), int(time.time()%100)))
    
if __name__ == '__main__':
  main()
