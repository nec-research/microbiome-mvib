"""
This script shows how the microbiome_mvib package can be used on custom data
to train the MVIB model for microbiome-based disease prediction.
"""
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
import torch
import sys
sys.path.insert(0, "../")

from microbiome_mvib.mvib import MVIB
from microbiome_mvib.dataset import CustomDataset
from microbiome_mvib.trainer import Trainer


# instantiate dataset object
data_dir = 'data/custom_data'
device = 'cpu'
dataset = CustomDataset(data_dir, 'CustomDisease', device, scale=True)

# instantiate MVIB
model = MVIB(
    n_latents=256,
    abundance_dim=len(dataset[0][0]),
    marker_dim=len(dataset[0][1]),
    device=device
).to(device)

# instantiate trainer
trainer = Trainer(
    model=model,
    epochs=20,
    lr=1e-5,
    beta=1e-5,
    lambda_abundance=0,
    lambda_markers=0,
    lambda_bce=1,
    lambda_triplet=1,
    checkpoint_dir='results',
    monitor='max'
)

# split
train_ids, test_ids, y_train, y_test = dataset.train_test_split(0.2, random_state=42)
inner_train_ids, val_ids, _, __ = model_selection.train_test_split(
    train_ids,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# create train loader
train_sampler = SubsetRandomSampler(inner_train_ids)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    sampler=train_sampler
)
# create val loader
val_sampler = SubsetRandomSampler(val_ids)
val_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    sampler=val_sampler
)

# training
state = trainer.train(
    train_loader,
    val_loader,
    bce=True,
    triplet=False,
    autoencoding=False
)
model, val_best_roc_auc = trainer.load_checkpoint(state)

# test
model.eval()
prediction = model.classify(dataset.abundance[test_ids], dataset.markers[test_ids]).cpu().detach()
test_roc_auc = roc_auc_score(y_test, prediction)
print('Test ROC AUC: ', test_roc_auc)

# compute stochastic encodings of the patients
mu, logvar = model.infer(dataset.abundance, dataset.markers)
