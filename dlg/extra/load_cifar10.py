import os
import pickle
import numpy as np
from torchvision.datasets import CIFAR10

def load_CIFAR10(data_dir='./cifar-10-batches-py'):
    """
    Load the CIFAR-10 dataset from the local directory.

    Args:
        data_dir (str): Path to the directory containing CIFAR-10 dataset files.

    Returns:
        dataset (list of tuples): List of (image, label) tuples.
    """
    dataset = []
    for batch in range(1, 6):  # CIFAR-10 has 5 training batches
        file_path = os.path.join(data_dir, f'data_batch_{batch}')
        with open(file_path, 'rb') as f:
            batch_data = pickle.load(f, encoding='latin1')
            images = batch_data['data']
            labels = batch_data['labels']
            
            # Reshape the images to 32x32x3
            images = images.reshape(-1, 3, 32, 32)
            images = np.transpose(images, (0, 2, 3, 1))  # Change to HWC format
            
            for img, label in zip(images, labels):
                dataset.append((img, label))

    # Load the test batch
    test_file_path = os.path.join(data_dir, 'test_batch')
    with open(test_file_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        test_images = test_batch['data']
        test_labels = test_batch['labels']
        
        test_images = test_images.reshape(-1, 3, 32, 32)
        test_images = np.transpose(test_images, (0, 2, 3, 1))  # Change to HWC format
        
        for img, label in zip(test_images, test_labels):
            dataset.append((img, label))

    return dataset

# Example usage:
# cifar10_data = load_CIFAR10('/path/to/cifar-10-batches-py')