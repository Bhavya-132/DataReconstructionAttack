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

# main.py
from agem import AGEM

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--index', type=int, default=25,  # Changed default to int (was string)
                    help='the index for leaking images on CIFAR.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

# Create an instance of AGEM and pass DEVICE
agem = AGEM(buffer_size=1000, batch_size=32)

# FIX: Unpack dataset correctly
images, labels = download_cifar10('../cifar-10-batches-py')

img_index = args.index
img_index = min(img_index, len(images) - 1)  # Ensure the index is within range

tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# Unpack image and label correctly
image, label = images[img_index], labels[img_index]  # Separate the image and label

# print(f"Label type: {type(label)}")  # Should be a scalar value or a tensor
# print(f"Label value: {label}")  # Inspect the actual label

if isinstance(image, tuple):  
    image = image[0]  # Extract actual image from tuple


# print(f"Image type: {type(image)}")  # Should be <class 'numpy.ndarray'>
# print(f"Image shape: {image.shape}")  # Should be (H, W, C) for an image
# print(f"Image dtype: {image.dtype}")  # Should be uint8

# Convert the image from torch.Tensor to numpy array (convert from channels-first format)
image = image.permute(1, 2, 0).cpu().numpy()  # Converts (C, H, W) to (H, W, C)

# Convert the dtype from float32 to uint8 (required by Image.fromarray)
image = (255 * image).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

gt_data = tp(Image.fromarray(np.array(image))).to(device)
# If label is a tuple, access the correct element, usually index 0
label = label[1]
gt_label = torch.tensor(label).long().to(device)  # Convert to tensor and move to device

# Ensure gt_label has the correct shape (batch_size, 1) for one-hot encoding
gt_label = gt_label.view(-1, 1)  # Reshape the label to (1, 1) if it's a scalar

# FIX: Keep only one occurrence of gt_data assignment
if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)

gt_data = gt_data.view(1, *gt_data.size())

# Convert label to one-hot
gt_onehot_label = label_to_onehot(gt_label, num_classes=10)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import ResNet18, LeNet, weights_init
##net = LeNet().to(device)
net = ResNet18().to(device)

torch.manual_seed(1234)
net.apply(weights_init)
criterion = cross_entropy_for_onehot

# Compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# Generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn_like(gt_onehot_label, requires_grad=True).to(device)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

# Initialize memory buffer for A-GEM
memory_buffer = []
buffer_size = 1000  # Adjust based on your needs

def project_gradient(grad, mem_grad):
    """Project the gradient onto the subspace defined by the memory gradients."""
    grad = grad.flatten()
    mem_grad = mem_grad.flatten()
    if mem_grad.numel() == 0:
        return grad.view_as(grad)
    dot_product = torch.dot(grad, mem_grad)
    if dot_product < 0:
        grad -= (dot_product / torch.dot(mem_grad, mem_grad)) * mem_grad
    return grad.view_as(grad)

history = []
dummy_dy_dx = None
for iters in range(300):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        print(f"dummy_loss: {dummy_loss.item()}")  # Debugging

        # First backward pass with retain_graph=True to keep the graph
        dummy_loss.backward(retain_graph=True)

        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        ##print(f"dummy_dy_dx: {dummy_dy_dx}")  # Debugging

        # Get the gradients
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        # Debugging: Check if dummy_dy_dx is None or has the expected values
        if dummy_dy_dx is None:
            print("No gradients were computed!")
        else:
            print(f"dummy_dy_dx: {dummy_dy_dx}")  # Check the gradients

        # Check if gradients are computed
        if dummy_dy_dx is None:
            raise ValueError("No gradients computed. Check your loss function or model output.")

        # Project gradients using A-GEM
        if memory_buffer:
            mem_gradients = torch.cat([g for mem_grad in memory_buffer for g in mem_grad])
            dummy_dy_dx = [project_gradient(g, mem_gradients) for g in dummy_dy_dx]

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    
    # Update memory buffer
    flattened_gradients = torch.cat([g.detach().clone().flatten() for g in dummy_dy_dx])
    if len(memory_buffer) < buffer_size:
        memory_buffer.append(flattened_gradients)
    else:
        memory_buffer[np.random.randint(0, buffer_size)] = flattened_gradients

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
