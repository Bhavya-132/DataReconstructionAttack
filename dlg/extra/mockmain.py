# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
from PIL import Image
from cifarten import download_cifar10
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)
from utils import label_to_onehot, cross_entropy_for_onehot
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default=25,
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)
# dst = datasets.CIFAR100("~/.torch", download=True)
dst = download_cifar10('../cifar-10-batches-py')
tp = transforms.ToTensor()
tt = transforms.ToPILImage()
img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)
if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)
gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
#gt_label = gt_label.view(1, )
#gt_onehot_label = label_to_onehot(gt_label)
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)
plt.imshow(tt(gt_data[0].cpu()))
from models.vision import ResNet18, LeNet, weights_init
##net = LeNet().to(device)
net = ResNet18().to(device)
torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot
# compute original gradient
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))
# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
plt.imshow(tt(dummy_data[0].cpu()))
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
history = []
for iters in range(300):
    def closure():
        optimizer.zero_grad()
        dummy_pred = net(dummy_data)
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        return grad_diff
    optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))
plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')
plt.show()
# Add AGEM for training
class AGEM:
    def __init__(self, model, memory_data, memory_labels, optimizer):
        self.model = model
        self.memory_data = memory_data
        self.memory_labels = memory_labels
        self.optimizer = optimizer
    def compute_reference_grad(self):
        self.model.zero_grad()
        memory_pred = self.model(self.memory_data)
        memory_loss = F.cross_entropy(memory_pred, self.memory_labels)
        memory_grad = torch.autograd.grad(memory_loss, self.model.parameters(), retain_graph=True)
        return [g.detach().clone() for g in memory_grad]
    def project_gradient(self, gradients, reference_grad):
        dot_product = sum(torch.dot(g.flatten(), r.flatten()) for g, r in zip(gradients, reference_grad))
        if dot_product < 0:
            grad_norm_sq = sum((g.flatten() ** 2).sum() for g in reference_grad)
            scale = dot_product / grad_norm_sq
            for g, r in zip(gradients, reference_grad):
                g -= scale * r
    def step(self, data, labels):
        self.model.zero_grad()
        pred = self.model(data)
        loss = F.cross_entropy(pred, labels)
        gradients = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        reference_grad = self.compute_reference_grad()
        self.project_gradient(gradients, reference_grad)
        loss.backward()
        self.optimizer.step()
# Example usage:
memory_data = torch.randn(100, 3, 32, 32).to(device)  # Random memory data
memory_labels = torch.randint(0, 10, (100,)).to(device)  # Random memory labels
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
agem = AGEM(net, memory_data, memory_labels, optimizer)
# Training loop
data_loader = torch.utils.data.DataLoader(dst, batch_size=32, shuffle=True)
for epoch in range(10):
    for batch_data, batch_labels in data_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        agem.step(batch_data, batch_labels)
