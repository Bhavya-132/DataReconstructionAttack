import torch
import torch.nn.functional as F


#def label_to_onehot(target, num_classes=100):
 #   target = torch.unsqueeze(target, 1)
  #  onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
   # onehot_target.scatter_(1, target, 1)
    #return onehot_target

# def label_to_onehot(target, num_classes=10):
#     target = torch.unsqueeze(target, 1)
#     onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
#     onehot_target.scatter_(1, target, 1)
#     return onehot_target

def label_to_onehot(target, num_classes=10):
    # Ensure target is of shape (batch_size, 1)
    target = target.view(-1, 1)  # Reshape to (batch_size, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)  # Scatter the 1s into the correct positions
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

