## microbiome-mvib
Implementation of the Multimodal Variational Information Bottleneck (MVIB) for microbiome-based disease prediction, proposed in the PLOS Computational Biology paper [Microbiome-based disease prediction with multimodal variational information bottlenecks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010050)

<img src="https://github.com/nec-research/microbiome-mvib/blob/master/Fig1.png" width="40%">

## Installation
```microbiome_mvib``` can be installed as a Python package.

**(PyTorch 1.7, CUDA 10.2)**
```
git clone https://github.com/nec-research/microbiome-mvib.git
cd microbiome-mvib
pip install .
```

## Train MVIB on custom data
The (compressed) directory `data/custom_data` contains exemplary custom data:
* `marker_CustomDisease.txt` is a template for the strain-level marker profile;
* `abundance_CustomDisease.txt` is a template for the species-relative abundance profile.

The script [train_custom_dataset.py](scripts/train_custom_dataset.py) shows how to use the main classes of `microbiome_mvib` for training
MVIB on custom data. In the following, the usage of the main classes is documented.

**Instantiate a dataset object**
```python
from microbiome_mvib.dataset import CustomDataset

data_dir = 'data\custom_data'
disease = 'CustomDisease'
device = 'cpu'
dataset = CustomDataset(data_dir, disease, device, scale=True)
```
The `CustomDataset` class is implemented as a typical [PyTorch dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
* `data_dir`: the path of the directory in which the data is stored. It expects the structure of `data/custom_data`
* `disease`: the name of the dataset/disease, which is expected in the name of the .txt files, e.g. `disease = CustomDisease` for `abundance_CustomDisease.txt`
* `device`: a [PyTorch device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) on which the dataset is stored
* `scale`: allows to standardize abundance features by removing the mean and scaling to unit variance

**Instantiate MVIB model**
```python
from microbiome_mvib.mvib import MVIB

model = MVIB(
    n_latents=256,
    abundance_dim=len(dataset[0][0]),
    marker_dim=len(dataset[0][1]),
    device=device).to(device)
```
The MVIB class inherits from [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
* `n_latents`: the bottleneck dimension
* `abundance_dim` and `marker_dim`: dimension of the abundance and marker feature vectors
* `device`: a [PyTorch device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) on which the model is stored

**Instantiate trainer object**
```python
from microbiome_mvib.trainer import Trainer

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
```
* `model`: Torch MVIB model object
* `epochs`: max training epochs
* `lr`: learning rate
* `beta`: multiplier for KL divergence / ELBO
* `lambda_abundance`: multiplier for abundance reconstruction loss
* `lambda_markers`: multiplier for markers reconstruction loss
* `lambda_bce`: multiplier for binary cross-entropy loss
* `lambda_triplet`: multiplier for triplet loss
* `checkpoint_dir`: directory for saving model checkpoints
* `monitor`: `min` minimize loss; `max` maximize ROC AUC (for selecting best model checkpoint)

**Train/validation/test split**
```python
from sklearn import model_selection

train_ids, test_ids, y_train, y_test = dataset.train_test_split(
    0.2, random_state=42
)
    
inner_train_ids, val_ids, _, __ = model_selection.train_test_split(
    train_ids,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)
```

**Instantiate data loaders**
```python
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import  DataLoader

# train
train_sampler = SubsetRandomSampler(inner_train_ids)
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=train_sampler
)

# validation
val_sampler = SubsetRandomSampler(val_ids)
val_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=val_sampler
)
```

**Train MVIB**
````python
state = trainer.train(
    train_loader,
    val_loader,
    bce=True,
    triplet=False,
    autoencoding=False
)
model, val_best_roc_auc = trainer.load_checkpoint(state)
````
* `train_loader`: the training Torch data loader
* `val_loader`: the validation Torch data loader
* `bce`: whether to optimize binary cross-entropy or not
* `triplet`: whether to optimize triplet loss or not
* `autoencoding`: whether to optimize reconstruction loss of not

**Remark: Train a Multimodal Variational Autoencoder**

When `autoencoding=True`, the ELBO loss of the Multimodal Variational Autoencoder (MVAE)
is optimized as well. See paper by Wu and Goodman, NeurIPS 2018:
https://papers.nips.cc/paper/2018/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf

It is possible to train the model in a pure self-supervised fashion by setting 
`bce=False`, `triplet=False`, `autoencoding=True` in the `train()` method and 
`monitor=min` in the `Trainer` object:

```python
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
    monitor='min'
)
...
state = trainer.train(
    train_loader,
    val_loader,
    bce=False,
    triplet=False,
    autoencoding=True
)
```

**Test the model**
```python
model.eval()
prediction = model.classify(dataset.abundance[test_ids], dataset.markers[test_ids]).cpu().detach()
test_roc_auc = roc_auc_score(y_test, prediction)
```

**Compute stochastic encodings for all samples in the dataset**
```python
mu, logvar = model.infer(dataset.abundance, dataset.markers)
```
The stochastic encodings of the samples are modelled as multivariate Gaussian distributions:
* `mu` is a torch.Tensor representing the mean vectors of the stochastic encodings
* `logvar` is a torch.Tensor containing the log-variance vectors of such distributions 

**Compute saliency maps**
```python
from microbiome_mvib.saliency import Saliency

saliency = Saliency(dataset)

saliency.init()  # store gradients of the input tensors
saliency.update(prediction, dataset.labels_gpu)
saliency.stop()  # stop storing gradients of the input tensors

saliency.save(disease_dir='results')
```

## Reproduce paper results

We run experiments on CentOS Linux 8 with Python 3.6.
An NVIDIA TITAN RTX GPU was used for MVIB.

**Data**

Decompress `data.zip` to a `<ROOT>` directory.

**Binary-cross entropy loss**
```
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --ensemble --data default --root <ROOT> --modality <A or M or AM>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --ensemble --data joint --root <ROOT> --modality <A or M or AM>
```

**Binary-cross entropy loss + Triplet margin loss**
```
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data default --root <ROOT> --modality <A or M or AM>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data joint --root <ROOT> --modality <A or M or AM>
```

**Transfer learning (Binary-cross entropy loss + Triplet margin loss)**
```
python train_transfer_learning.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data joint --root <ROOT>
```

**Random Forest benchmark - results on single modalities and abundance+markers**
```
python rf_benchmark.py --modality <markers OR abundance OR both> --root <ROOT>
```

**Cross-study generalization results**
```
python cross_study.py --gpu 0 --lr 0.00001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --epochs 100 --root <ROOT>
```

**Trimodal experiments on Yachida et al. (PMID: 31171880) with metabolic profiles**
```
python train_metabolic.py --gpu 0 --lr 0.00001 --n-latents 128 --beta 0.000001 --ensemble --epochs 100 --root <ROOT>
```

**Random Forest benchmark - trimodal experiments on Yachida et al. (PMID: 31171880) with metabolic profiles**
```
python rf_yachida_metabolic.py --root <ROOT>
```

**Saliency**
```
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data default --saliency --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data joint --saliency --root <ROOT>
```

**Training time analysis**
```
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.000001 --bce --data default --dataset Colorectal-YachidaS --epochs 50
python train_metabolic.py --gpu 0 --lr 0.0001 --n-latents 256 --beta 0.000001 --epochs 50
python time_benchmark.py --modality <A or M or AM or AMM>
```

## Cite
If you find this repository useful, please cite our paper:
```
@article{10.1371/journal.pcbi.1010050,
    doi = {10.1371/journal.pcbi.1010050},
    author = {Grazioli, Filippo AND Siarheyeu, Raman AND Alqassem, Israa AND Henschel, Andreas AND Pileggi, Giampaolo AND Meiser, Andrea},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Microbiome-based disease prediction with multimodal variational information bottlenecks},
    year = {2022},
    month = {04},
    volume = {18},
    url = {https://doi.org/10.1371/journal.pcbi.1010050},
    pages = {1-27},
    number = {4},
}
```
