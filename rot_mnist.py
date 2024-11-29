import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import sys
import time

from torch.utils.data import TensorDataset, Subset, DataLoader, random_split

from src.models import MLP, Standardize
from src.liegg import polarization_matrix_2, symmetry_metrics

def rotate_resample(img, G):
    
    # img: (1, 1, 28, 28)
    # G: (2,2)
    
    # proc input image
    img_in = img.cpu().data
    
    # wrap to affine transform
    R = torch.zeros((2,3))
    R[:2,:2] = G.cpu().data
    
    # sample grid
    grid = nn.functional.affine_grid(R.unsqueeze(0), img_in.size())
    
    # resample img
    img_out = nn.functional.grid_sample(img_in, grid)
    
    return img_out[0,0]

def get_random_subset_bool_mask(tensor, num_samples):
    """Using boolean mask"""
    N = tensor.size(0)
    mask = torch.zeros(N, dtype=torch.bool)
    indices = torch.randperm(N)[:num_samples]
    mask[indices] = True
    return tensor[mask]


def split_tensor_data(dataset, splits, seed=2022):
    """
    Split a tensor into train/test/val datasets
    
    Args:
        tensor: torch.Tensor of shape (N, d)
        splits: dict with keys ('train', 'test', 'val') and values (counts)
        seed: random seed for reproducibility
    
    Returns:
        dict with keys ('train', 'test', 'val') containing Subset objects
    """
    # Create a TensorDataset from the input tensor    
    # Process splits similar to original function
    split_values = np.array(list(splits.values()))
    assert (split_values == -1).sum() <= 1, "dict(splits) permits only one dynamic argument"
    
    # Calculate length for dynamic split (-1)
    off_len = len(dataset) - split_values[split_values != -1].sum()
    split_values[split_values == -1] = off_len
    
    # Perform random split
    splitted = torch.utils.data.random_split(dataset, 
                                           split_values,
                                           generator=torch.Generator().manual_seed(seed))
    
    # Record to dict with same keys as input splits
    out_data = {}
    for i, each_k in enumerate(splits.keys()):
        out_data[each_k] = splitted[i]
    
    return out_data


class RotatedMNIST(Dataset):
    def __init__(self, root="./data", train=True, download=True, angles=None):
        """
        Custom Dataset for Rotated MNIST that applies rotations on-the-fly.
        
        Args:
            root (str): Path to store the MNIST data
            train (bool): If True, creates dataset from training set
            download (bool): If True, downloads the dataset
            angles (list): List of rotation angles in degrees. If None, uses [0]
        """
        self.angles = angles if angles is not None else [0]
        
        # Load regular MNIST
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])
        )
        
    def __len__(self):
        return len(self.mnist) * len(self.angles)
    
    def __getitem__(self, idx):
        """
        Returns a rotated MNIST digit.
        
        The index is mapped to both the original image index and the rotation angle.
        """
        # Map idx to original image index and angle index
        mnist_idx = idx // len(self.angles)
        angle_idx = idx % len(self.angles)
        angle = self.angles[angle_idx]
        
        # Get original image and label
        image, label = self.mnist[mnist_idx]
        
        # Apply rotation
        if angle != 0:
            image = transforms.functional.rotate(image, angle)
            
        return image.flatten(), torch.tensor(label)

# Example usage and dataloader creation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create dataset with multiple rotation angles
    angles = [10 * i for i in range(36)]
    dataset = RotatedMNIST(
        root="./rot_mnist/data",
        train=True,
        download=True,
        angles=angles
    )

    print(len(dataset))

    train_dataset, test_dataset = random_split(
        dataset,
        [1500000, len(dataset) - 1500000],  # Specific numbers of samples for train/test
        generator=torch.Generator().manual_seed(42)
    )

    model = MLP(in_dim = 28*28, out_dim=10, ch=256, num_nonlins=4)
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-3, weight_decay=1e-4)
    loss = torch.nn.CrossEntropyLoss()   

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16384,
            shuffle=True,
            num_workers=4
        ),
        'test': torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16384,
            shuffle=False,
            num_workers=4
        )
    }

    # Move model to GPU
    model = model.cuda()

    # Before training test
    losses = []
    for (x_test, y_test) in dataloaders['test']:
        x_test, y_test = x_test.cuda(), y_test.cuda()  # Move to GPU
        y_pred = model(x_test.float())
        before_train = loss(y_pred, y_test)
        losses.append(before_train.cpu().detach().numpy())  # Move back to CPU for numpy

    print("losses before: ", np.mean(losses))

    # Training loop
    n_epochs = 1
    losses = []
    for epoch in range(n_epochs):
        running_loss = 0
        for i, (inputs, labels) in enumerate(dataloaders['train'], 0):
            # Move batch to GPU
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs.float())

            loss_ = loss(outputs, labels)
            losses.append(loss_.cpu().detach())  # Move to CPU for storage
            loss_.backward()
            optimizer.step()

            running_loss += loss_
            if i % 10 == 0:
                sys.stdout.write('[%d, %5d] loss: %.3f\r' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    plt.plot(losses)
    plt.savefig("loss_curve_mnist.png")
    plt.clf()

    losses = []
    for (x_test, y_test) in dataloaders['test']:
        pol = x_test.float()
        x_test, y_test = x_test.cuda(), y_test.cuda()  # Move to GPU
        y_pred = model(x_test.float())
        after_train = loss(y_pred, y_test)
        losses.append(after_train.cpu().detach().numpy())  # Move back to CPU for numpy

    print("losses after: ", np.mean(losses))

    pol = pol.reshape((pol.shape[0], 28, 28))

    model = model.cpu()
    
    E = polarization_matrix_2(model, pol)
    
    singular_values, symmetry_biases, generators = symmetry_metrics(E)

    img = pol[111].float()
    plt.imsave("img.png", img)

    for index, generator in enumerate(generators):
        img_r = rotate_resample(img.reshape(1, 1, *pol[1].shape), generator)
        plt.imsave("generator " + str(index) + ".png", img_r)

    # # Create DataLoader
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=256,
    #     shuffle=True,
    #     num_workers=4
    # )
    
    # # Visualize some examples
    # def show_examples(dataloader, num_examples=5):
    #     images, labels = next(iter(dataloader))
    #     plt.figure(figsize=(15, 3))
    #     for i in range(num_examples):
    #         plt.subplot(1, num_examples, i + 1)
    #         plt.imshow(images[i].squeeze(), cmap='gray')
    #         plt.title(f'Label: {labels[i]}')
    #         plt.axis('off')
    #     plt.savefig("rot_mnist.png")
    
    # for X_batch, y_batch in dataloader:
    #     print(X_batch.shape)
