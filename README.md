🦔 HEDGEhog 🦔: BERT-based multi-class uncertainty cues recognition
===========================================================

# Description
The repo contains code for fine-tuning a pretrained language model (BERT, SciBERT, etc.) for the task of multi-class classification of uncertainty cues (a.k.a *hedges*).

You can use the code to either:

- train and evaluate your own model (see [Train and evaluate](#train-and-evaluate)); or
- use my fine-tuned model to generate predictions on your data (see [Predict](#predict)).

I use the [Simple Transformers](https://simpletransformers.ai/) and [W&B](https://docs.wandb.ai/) packages to perform the fine-tuning.

# Contents

1. [Setup](#setup)
2. [Data](#data)
3. [Performance](#performance)
4. [Usage](#usage)  
    4.1 [Train and evaluate](#train-and-evaluate)  
    4.2 [Predict](#predict)
5. [References](#references)

# Setup
The requirements are listed in the [environment.yml](environment.yml) file. It is recommended to create a virtual environment with conda (you need to have `Anaconda` or `Miniconda` installed):
```
$ conda env create -f environment.yml
$ conda activate hedgehog
```

# Data
HEDGEhog is trained and evaluated on the [Szeged Uncertainty Corpus](https://rgai.inf.u-szeged.hu/node/160) (Szarvas et al. 2012<sup>1</sup>). The original sentence-level XML version of this dataset is available [here](https://rgai.inf.u-szeged.hu/node/160).

The token-level version that is used in the current repo can be downloaded from [here](https://1drv.ms/u/s!AvPkt_QxBozXk7BiazucDqZkVxLo6g?e=IisuM6) in a form of pickled pandas DataFrame's. You can download either the split sets (```train.pkl``` 137MB, ```test.pkl``` 17MB, ```dev.pkl``` 17MB) or the full dataset (```szeged_fixed.pkl``` 172MB).

Each row in the df contains a token, its features (these are not relevant for HEDGEhog; they were used to train the baseline CRF model, see [here](https://github.com/vanboefer/uncertainty_crf)), its sentence ID, and its label. The labels refer to different types of semantic uncertainty (Szarvas et al. 2012) -

- **E**pistemic: the proposition is possible, but its truth-value cannot be decided at the moment. Example: *She **may** be already asleep.*
- **I**nvestigation: the proposition is in the process of having its truth-value determined. Example: *She **examined** the role of NF-kappaB in protein activation.*
- **D**oxatic: the proposition expresses beliefs and hypotheses, which may be known as true or false by others. Example: *She **believes** that the Earth is flat*
- Co**N**dition: the proposition is true or false based on the truth-value of another proposition. Example: ***If** she gets the job, she will move to Utrecht.*
- **C**ertain: the token is not an uncertainty cue.

# Performance
Here is the performance of my [downloadable fine-tuned model](https://1drv.ms/u/s!AvPkt_QxBozXk7xX29OAFO5JLuftwQ?e=f6ABI0) on the test set:

class | precision | recall | F1-score | support
---|---|---|---|---
Epistemic | 0.90 | 0.85 | 0.88 | 624
Doxatic | 0.88 | 0.92 | 0.90 | 142
Investigation | 0.83 | 0.86 | 0.84 | 111
Condition | 0.85 | 0.87 | 0.86 | 86
Certain | 1.00 | 1.00 | 1.00 | 104,751
**macro average** | **0.89** | **0.90** | **0.89** | 105,714

# Usage

## Train and evaluate
You can use the data and the code to train your own model, for example with another pretrained language model as basis or with different hyperparameters. To do this, follow the following steps:

1. Download the [data](https://1drv.ms/u/s!AvPkt_QxBozXk7BiazucDqZkVxLo6g?e=IisuM6) and place `train.pkl`, `test.pkl`, `dev.pkl` in the ```data/``` directory.
2. Add a dictionary with your new model args to the [config.json](src/config.json) file. See [Simple Transformers](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model) for all the possible configuration options.
3. Adjust the `--model_args`, `--model_type` and `--model_name` parameters in [train_model.py](src/train_model.py). You can either change the default values in the script or pass your arguments in the command line; for example -
```
$ python train_model.py --model_args my_new_args --model_type roberta --model_name roberta-base
```

To evaluate your model, use the [evaluate_model.py](src/evaluate_model.py) script. Adjust the `--model_type` and `--model_name` parameters for your trained model, set the `--output` parameter to the path where you want to save the pickled model predictions. You can adjust the [evaluate_model.py](src/evaluate_model.py) script to add additional evaluation metrics; see the docstring in the file and the [Simple Transformers](https://simpletransformers.ai/docs/tips-and-tricks/#additional-evaluation-metrics) documentation for more details.

You can perform a sweep for hyperparameters optimization with the [wandb_sweep.py](src/wandb_sweep.py) script. See the docstring in the file, [Simple Transformers](https://simpletransformers.ai/docs/tips-and-tricks/#hyperparameter-optimization) documentation and [W&B](https://docs.wandb.ai/) documentation for more details.

## Predict
To use my fine-tuned model for generating predictions on your own data, follow the following steps:
1. Prepare your data in a pickled DataFrame which contains the column 'sentence'. For each row in the df, the text in 'sentence' will be split on space and a label will be predicted for each token. A list with the predicted labels will be saved in a new column named 'predictions'.
2. Download the `hedgehog` folder from [here](https://1drv.ms/u/s!AvPkt_QxBozXk8Y4aWzTOVoUjcqSVw?e=B6BxPZ) and place it in the ```models/``` directory. The folder contains the model `pytorch_model.bin` and info about the tokenizer, the vocabulary and the configuration.
3. Run the [predict.py](src/predict.py) script, indicating the path to your pickled data (alternatively, edit the default value in the script):
```
$ python predict.py --data_pkl ../data/mydata.pkl
```

# References
<sup>1</sup> Szarvas, G., Vincze, V., Farkas, R., Móra, G., & Gurevych, I. (2012). Cross-genre and cross-domain detection of semantic uncertainty. *Computational Linguistics, 38*(2), 335-367.
