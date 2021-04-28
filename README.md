# Model-based Data-Complexity Estimator for Deep Learning Systems

## About
This repository provides source codes to reproduce experiments described in [Model-based Data-Complexity Estimator for Deep Learning Systems](./docs/aitest.pdf).

## How to Use
### Set up
1. [Install poetry](https://python-poetry.org/docs/) 
1. Install dependencies
    ```bash 
    $ poetry install 
    ```

### Calculate complexities of datasets based on your model
1. Run the following command
    ```bash
    $ poetry shell
    $ python ./onestop.py [path_to_your_model] [path_to_training_dataset] [layer_name] (--test [path_to_test_dataset]) (--tag [tag_to_identify_results])
    ```
1. The calculated complexities are saved in ``results'' folder

### Reproduce results in the paper
1. Run the following commannd
    ```bash
    $ cd experiments
    $ poetry install
    $ poetry shell
    $ jupyter notebook
    ```
1. Open jupyter notebook and run cells in order.

## Folder Tree
<pre>
.
├─datasets
│  ├─test
│  └─training
├─docs
├─experiments
│  ├─activation_traces
│  │  ├─test
│  │  └─training
│  ├─nmf_info
│  │  ├─base_weight
│  │  └─model
│  ├─pred_res
│  └─results
├─intermediate_results
│  ├─activation_traces
│  └─nmf_information
├─src
│  └─lib
└─trained_models
</pre>

### datasets
Datasets used in the experiments (.npz files)

### docs
Files for [GitHub Pages showing supplemental materials](https://nttdata-rdh.github.io/complexity/)

### experiments
Resource to conduct experiments in the paper.
#### activation_traces
Activation Traces used in the experiments.  
Since the file size is too large, we just show hash values of the files in this repo.
If you need the original files, please contact us.

#### nmf_information
- base_weight  
  Feature Matrix obtained in the experiments (.npz files)
- model  
  NMF that fit to the activation traces of inputs in each training dataset (.pkl files)

### intermediate_results
AT and NMF information is stored if you calculate complexities on your model and datasets.

### src
Source codes to calculate complexities.

### trained_models
Trained model used in the experiments (.h5 files)