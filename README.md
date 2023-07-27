# Computational Optimization of DNA Activity (CODA)
For legacy reasons, this repo is currently called boda2

## Contents

## Overview
Here, we present a platform to engineer and validate synthetic CREs capable of driving gene expression
with programmed cell type specificity. This library contains the resources needed to train DNNs on MPRA data and generate synthetic sequences from these models. Additionally, we include resources to apply these model to to inference tasks on common input data types (e.g., VCF, FASTA, etc.). Finally, we provide examples which deploy the Malinois model.

## System requirements
### Hardware requirements
CODA was extensively tested in Google Colab environments and GCP VMs with the following specs:

Type: `a2-highgpu-1g`
CPU: 12 vCPU
RAM: 85 GB
GPU: 1x Tesla A100
GPU-RAM: 40 GB HBM2

### Software requirements
CODA was designed using the GCP deployment: NVIDIA GPU-Optimized Image for Deep Learning, ML & HPC

OS: Ubuntu 20.04.2 LTS
CUDA: 11.3

## Installation Guide
CODA can be installed from the latest version of the GITHUB repo.
```
git clone https://github.com/sjgosai/boda2.git
cd boda2/
pip install -e .
```

## Interactive Docker Environments
CODA has been installed in docker containers which can be downloaded in attached for interactive use. This can be quickly deployed using helper scripts in the repo:
```
cd /home/ubuntu
git clone https://github.com/sjgosai/boda2.git
bash boda2/src/run_docker_for_dev.sh gcr.io/sabeti-encode/boda devenv 0.2.0 8888 6006
```
Which connects `jupyter lab` to ports `8888` and `6006`.

## Interactive modeling and deployment
CODA is an extension of pytorch and pytorch-lightning. Classes in CODA generally inherit from `nn.Module` and `lightning.LightningModule` but need to be combined as described in `./boda2/boda/README.md`.

Example interactive deployment of Malinois can be found here: `./boda2/analysis/SG016__inference_package_dev/basic_load_model.ipynb`

## Applications
We have developed python applications to train models and generate sequences using software implmentations in this library.

### Model training
Deep learning models can be trained from the command line by invoking the DATA, MODEL, and GRAPH modules. For example:
```
python main.py --data_module SOMETHING --model_module SOMETHING --graph_module SOMETHING ...
```

### Sequence design
Trained models can be deployed to generate sequences using implemented algorithms:
```
python generate.py \
  --params_module StraightThroughParameters \
    --batch_size 512 --n_channels 4 --length 200 --n_samples 10 \
  --energy_module OverMaxEnergy \
    --model_artifact gs://syrgoth/aip_ui_test/model_artifacts__20211113_021200__287348.tar.gz \
    --bias_cell 0 --bending_factor 1.0 --a_min -2.0 --a_max 6.0 \
  --generator_module FastSeqProp \
    --energy_threshold -2.0 --max_attempts 20 --n_steps 200 \
    --n_proposals 2000 \
  --proposal_path gs://syrgoth/boda_library_design_202112/sg__k562__fsp__test
```

### Variant effect prediction
Trained models can be deployed to infer the effect of non-coding variants in CREs
```
python vcf_predict.py \
  --artifact_path gs://jax-tewhey-boda-project-data/common_data/model_artifacts__20211113_021200__287348.tar.gz \
  --vcf_file hepg2.ase.calls.fdr.05.vcf \
  --fasta_file GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
  --output test.vcf \
  --relative_start 25 --relative_end 180 --step_size 25 \
  --strand_reduction mean --window_reduction gather \
  --window_gathering abs_max --gather_source skew --activity_threshold 0.5 \
  --batch_size 16
```

### Saturation mutagenesis
UNDER CONSTRUCTION

## Extending CODA
CODA is modular. If new modules fit the API requirements, they will work with the entire system, including deployment applications.

## Cloud integrations
Containerized CODA applications can be used in combination with various GCP platforms.

### Training models on VertexAI
UNDER CONSTRUCTION

### Deploying inference with Life Sciences API
UNDER CONSTRUCTION
