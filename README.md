# Informative Subgraph Extraction with Deep Reinforcement Learning for Drug-Drug Interaction Prediction



# Installation & Dependencies

RISE-DDI is mainly tested on a Linux OS with NVIDIA A100 40G and its dependecies are below.

|Package|Version|
|-----:|-------|
|python| 3.6.13|
|rdkit|2021.9.4|
|pytorch| 1.10.2|
|cuda|11.6.2|
|torch-cluster|1.5.9|
|torch-geometric| 2.0.2|
|torch-scatter |2.0.9|
|torch-sparse| 0.6.12|
|torchvision| 0.11.3|


# Datasets

Extract data.tar.gz and place it in the home directory

# Train

RISE-DDI can be trained with the following command:

+ drugbank - transductive
```
python -u main.py --dataset drugbank --extractor RL --epoch 100 --sampler_lr 0.001 --layer 1 --khop 1 --pos 2 --neg 1  --k_step 60 --batch_size 32
```

+ kegg - transductive
```
python -u main.py --dataset kegg --extractor RL --epoch 100 --sampler_lr 0.001 --layer 1 --khop 1 --pos 2 --neg 1 --k_step 40 --batch_size 32 --eps 3e-6 --fixed_num 16
```

+ obgl-biokg - transductive
```
python -u main.py --dataset ogbl-biokg --extractor RL --epoch 100 --sampler_lr 0.001 --layer 1 --khop 1 --pos 2 --neg 1 --k_step 60 --batch_size 32 --eps 3e-6
```

+ drugbank - inductive
```
python -u main.py --dataset drugbank --extractor RL --epoch 100 --sampler_lr 0.001 --layer 1 --khop 1 --pos 2 --neg 1 --mode s4 --k_step 5 --fixed_num 1  --s_type inductive
```

+ kegg - inductive
```
python -u main.py --dataset kegg --extractor RL --epoch 100 --sampler_lr 0.001 --layer 1 --khop 1 --pos 2 --neg 1 --mode s4 --k_step 5 --fixed_num 1 --eps 3e-6 --s_type inductive
```

+ obgl-biokg - inductive
```
python -u main.py --dataset ogbl-biokg --extractor RL --epoch 100 --sampler_lr 0.001 --layer 1 --khop 1 --pos 2 --neg 1 --mode s4 --k_step 5 --fixed_num 1 --eps 3e-6 --s_type inductive
```

