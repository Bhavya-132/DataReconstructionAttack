import torchvision
import torchvision.transforms as transforms
import torch
import os
import ssl
import torchvision
#torch.Size([3, 32, 32])
def download_cifar10(data_dir):
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    print("Download complete.")
    return trainset, testset
def verify_cifar10(trainset, testset):
    """Verify the downloaded CIFAR-10 dataset by checking its size and a sample."""
    print("\nVerifying CIFAR-10 dataset...")
    # Check the size of the train and test datasets
    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of test samples: {len(testset)}")
    # Display a random sample image from the training dataset
    sample_idx = torch.randint(0, len(trainset), size=(1,)).item()
    image, label = trainset[sample_idx]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample label: {label}")
    # Ensure labels are within valid range (0-9 for CIFAR-10)
    train_labels_valid = all(0 <= label < 10 for _, label in trainset)
    test_labels_valid = all(0 <= label < 10 for _, label in testset)
    if train_labels_valid and test_labels_valid:
        print("\nVerification successful: All labels are within valid range.")
    else:
        print("\nVerification failed: Labels out of range detected.")
if __name__ == "__main__":
    # Directory to store the CIFAR-10 dataset
    data_directory = "./data"
    # Create directory if it doesn't exist
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    # Download CIFAR-10 dataset
    train_data, test_data = download_cifar10(data_directory)
    # Verify the dataset
    verify_cifar10(train_data, test_data)