Self-Attentive Ensemble Transformer
===================================
Representing Ensemble Interactions in Neural Networks for Earth System Models
---

![front_image](https://user-images.githubusercontent.com/17099005/122655730-05859d00-d155-11eb-9096-db106b34b3fe.png)

If you are using these scripts and the repository, please cite:

> Tobias Sebastian Finn, 2021. Self-Attentive Ensemble Transformer: 
> Representing Ensemble Interactions in Neural Networks for Earth System 
> Models. ArXiv: https://arxiv.org/abs/2106.13924
--------

This project and repository is dedicated to enable processing of ensemble data 
for Earth system models with neural networks. 
Based on ideas from self-attention and ensemble data assimilation, specifically
the ensemble Kalman filter, this repostory includes modules for the 
self-attentive ensemble transformer.
The ensemble transformer is a novel type of neural network to process 
ensemble data without a parametric assumption, as it is usually done in 
post-processing or Model Output Statistics.
With this repo, it is possible to compare the transformer with other models, 
like a parametric approach similar to [[Rasp & Lerch 2018]](#rasp).

The scripts and module is written in PyTorch [[1]](#1), Pytorch lightning [
[2]](#2) and 
configured with Hydra [[3]](#3).

The folder structure is the following:
```
.
|-- configs            # The hydra config scripts.
|-- data               # Storage of the data
|   |-- interim        # Data that is included within the repo
# The experiment data (the used configs for all experiments can be found in 
# sub directories and the hydra folder.
|   |-- models         # The trained models will be stored here
|   |-- processed      # The processed data is stored here
|   |   |-- era5       # The processed ERA5 data as zarr directories
|   |   |-- ifs        # The processed IFS data as zarr directories
|   |   |-- predicted  # The post-processed network predictions as NetCDF-files
|   |-- raw            # The raw data should be stored here
|   |   |-- era5       # The raw ERA5 data as NetCDF-files
|   |   |-- ifs        # The raw IFS data as NetCDF-files
|   |-- tensorboard    # Data for tensorboard visualization can be stored here
|-- ens_transformer    # The python module with different model etc.
|   |-- layers         # Different PyTorch modules for ensemble processing
|   |-- models         # Pytorch Lightning network specifications
|   |-- transformers   # Different self-attentive transformer modules
|-- notebooks          # Notebooks that were used to visualize the results
|-- scripts            # The scripts that were used to train the models
|   |-- data           # Scripts and notebooks to download and process the data
|   |-- predict.py     # Script to predict with a trained neural network
|   |-- train.py       # Hydra-script to train the networks
|-- environment.yml    # A possible environment configuration
|-- LICENSE            # The license file
|-- README.md          # This readme
|-- setup.py           # The setup.py to install the ens_transformer modules
|-- used_env.yml       # The used conda environment with pinned versions
```
In almost all scripts only relative directories are used to reference the 
data and models.

As a first step the ERA5 data has to be downloaded from [[4]](#4). All other 
scripts to pre-process the data can be found in `scripts/data/`.
The data raw model data to be put into `scripts/data/raw`.

Afterwards, the `scripts/train.py` script can be used to train the networks.
Specific options can be overwritten via Hydra syntax [[3]](#3).
To reproduce networks from the paper different model configurations are 
stored under `data/models/*/hydra/config.yaml`.
These files can be then used to rerun the experiment.
The subfolder `data/models/subsampling` was used for the subsampling experiments.
The subfolder `data/models/baseline_scaling` was used for the scaling 
experiments with the baseline models.
The subfolder `data/models/transformer_scaling` was used for the scaling 
experiments with the transformer networks.

The front image shows the attention map of one single attention head within 
the first layer of the transformer scaling experiment with 5 attention 
layers for 2019-09-01 12:00 UTC. Red colors indicate regions with high 
importance for the attention, whereas blueish colors show regions with low and
negative importance for the attention. This particular attention head is 
apparently activated by regions below the freezing level.

If you have further questions, please feel free to contact me or to create a 
GitHub issue.

--------
## References
<a id="1">[1]</a> https://pytorch.org/

<a id="2">[2]</a> https://www.pytorchlightning.ai/

<a id="3">[3]</a> https://hydra.cc/

<a id="4">[4]</a> https://cds.climate.copernicus.eu/

<a id="rasp">[Rasp & Lerch 2018]</a> Rasp, Stephan, and Sebastian Lerch. 
"Neural Networks for Postprocessing Ensemble Weather Forecasts", Monthly 
Weather Review 146, 11 (2018): 3885-3900,
https://doi.org/10.1175/MWR-D-18-0187.1


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
