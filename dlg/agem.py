# agem.py
import torch
import numpy as np

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

#AGEM Implementation
class AGEM:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = []
        self.task_counter = 0
        self.device = device  # Store the device

    def update_memory(self, data_loader):
        all_data = [(img, lbl) for img, lbl in data_loader.dataset]
        np.random.shuffle(all_data)
        memory_size_per_task = self.buffer_size // (self.task_counter + 1)
        self.memory = all_data[:memory_size_per_task]

    def sample_memory(self):
        if len(self.memory) == 0:
            return None
        memory_indices = np.random.choice(len(self.memory), size=self.batch_size, replace=True)
        memory_batch = [self.memory[idx] for idx in memory_indices]
        images, labels = zip(*memory_batch)
        return torch.stack(images).to(self.DEVICE), torch.tensor(labels).to(self.DEVICE)
