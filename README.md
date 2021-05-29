# Microbiome-based Disease Prediction with Multimodal Variational Information Bottlenecks

## Data
Decompress `data.zip` to a `<ROOT>` directory.

## Environment setup
We run experiments on CentOS Linux 8 with Python 3.6.

**Anaconda (CUDA 10.2)**
```
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

**pip (CUDA 10.1)**
```
pip install -r requirements.txt
pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Reproduce paper results

**Binary-cross entropy loss**
```
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data filtered0.1 --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data filtered0.8 --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data default --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data joint --root <ROOT>
```

**Binary-cross entropy loss + Triplet margin loss**
```
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --ensemble --data filtered0.1 --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --ensemble --data filtered0.8 --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --ensemble --data default --root <ROOT>
python train.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --ensemble --data joint --root <ROOT>
```

**Transfer learning (Binary-cross entropy loss + Triplet margin loss)**
```
python train_transfer_learning.py --gpu 0 --lr 0.0001 --n-latents 256 --no-autoencoding --lambda-abundance 0 --lambda-markers 0 --beta 0.00001 --bce --triplet --ensemble --data joint --root <ROOT>
```
