import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import sys
import time

from torch.utils.data import TensorDataset, Subset, DataLoader

from src.models import MLP, Standardize
from src.liegg import polarization_matrix_2, symmetry_metrics

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


model = MLP(in_dim = 64*64, out_dim=5, ch=128, num_nonlins=3)
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-3, weight_decay=1e-4)
loss = torch.nn.MSELoss()


path = '/home/sbharthulwar/thesis/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

data = np.load(path, allow_pickle=True, encoding='latin1')
imgs = data["imgs"]
latents_classes = data["latents_classes"]

latents_classes = torch.tensor(latents_classes)

ysubset = torch.cat([latents_classes[:, :3], latents_classes[:, 4:]], dim=1)  

subset = imgs

######################################

before = torch.tensor(subset)

subset = subset.reshape(subset.shape[0], 64*64)

subset = torch.tensor(subset)

ysubset = torch.tensor(ysubset)

dataset = TensorDataset(subset, ysubset)

datasets = split_tensor_data(dataset, {'train' : 637000, 'val' : 280, 'test' : 100000})

training_set_size=637000

n_epochs = int(900000/training_set_size)
batch_size = 1000
bs = batch_size
dataloaders = {k:DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
            num_workers=0,pin_memory=False) for k,v in datasets.items()}
dataloaders['Train'] = dataloaders['train']

for (x_test, y_test) in dataloaders['test']:
    y_pred = model(x_test.float())
    after_train = loss(y_pred, y_test.float()) 
    print('Test loss before Training' , after_train.item())

for epoch in range(n_epochs):
    running_loss = 0
    for i, (inputs, labels) in enumerate(dataloaders['Train'], 0):
        optimizer.zero_grad()
        outputs = model(inputs.float())

        loss_ = loss(outputs, labels.float())
        loss_.backward()
        optimizer.step()

        running_loss += loss_
        if i % 10 == 0:
            sys.stdout.write('[%d, %5d] loss: %.3f\r' %
                (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0


for (x_test, y_test) in dataloaders['test']:
        y_pred = model(x_test.float())
        after_train = loss(y_pred, y_test.float()) 


print("SUBSET FLOAT SHAPE", before.float().shape)

pol = get_random_subset_bool_mask(before.float(), 2000)

E = polarization_matrix_2(model, pol)

print(E.shape)
if not E.any():
    print("all zeros")

singular_values, symmetry_biases, generators = symmetry_metrics(E)

print(generators)

################################################################################


print('Symmetry variance: ', singular_values[-1].item())
print('min Symmetry bias: ', symmetry_biases[-1].item())

fig, ax = plt.subplots(1, 2, figsize=(11, 3))

ax[0].grid(axis='y')
ax[0].plot(singular_values.data, color='black')
ax[0].scatter(torch.arange(singular_values.shape[0]), singular_values.data, color='black', linewidths=.5)
ax[0].set_yscale('log')
ax[0].set_title('Polarization spectrum')
ax[0].set_xlabel('i-th singular value')

ax[1].grid(axis='y')
ax[1].plot(symmetry_biases.data, color='black')
ax[1].scatter(torch.arange(symmetry_biases.shape[0]), symmetry_biases.data, color='black', linewidths=.5)
ax[1].set_title('O(5) symmetry bias')
ax[1].set_xlabel('i-th singular vector')

plt.savefig('figure.png')