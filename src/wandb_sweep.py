"""
Perform a sweep for hyperparameters optimization, using Simple Transformers and W&B Sweeps.
The sweep is configured in a dictionary in a config file, which should specify the search strategy, the metric to be optimized, and the hyperparameters (and their possible values).
"""


import argparse
import warnings
import json
import torch
import wandb
import pandas as pd
from simpletransformers.ner import NERModel


def main(
    train_pkl,
    eval_pkl,
    config_json,
    sweep_config,
    model_args,
    model_type,
    model_name,
):
    """
    Perform a sweep for hyperparameters optimization, using Simple Transformers and W&B Sweeps.
    The sweep is configured in a dictionary in `config_json`, which should specify the search strategy, the metric to be optimized, and the hyperparameters (and their possible values).

    Parameters
    ----------
    train_pkl: str
        path to pickled df with the training data, which must contain the columns 'sentence_id', 'words' and 'labels'
    eval_pkl: str
        path to pickled df for evaluation during training
    config_json: str
        path to a json file containing the sweep config
    sweep_config: str
        the name of the sweep config dict from `config_json` to use
    model_args: str
        the name of the model args dict from `config_json` to use
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        the exact architecture and trained weights to use; this can be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model file

    Returns
    -------
    None
    """

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn('CUDA device not available; running on a CPU!')

    # load data
    train_data = pd.read_pickle(train_pkl)
    eval_data = pd.read_pickle(eval_pkl)

    # sweep config & model args
    with open(config_json, 'r') as f:
        config = json.load(f)
    sweep_config = config[sweep_config]
    model_args = config[model_args]

    sweep_id = wandb.sweep(sweep_config, project=model_args['wandb_project'])

    def train():
        wandb.init()

        model = NERModel(
            model_type,
            model_name,
            args=model_args,
            use_cuda=cuda_available,
            sweep_config=wandb.config,
        )

        model.train_model(train_data, eval_data=eval_data)

        wandb.join()

    wandb.agent(sweep_id, train)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_pkl', default='../data/train.pkl')
    argparser.add_argument('--eval_pkl', default='../data/dev.pkl')
    argparser.add_argument('--config_json', default='config.json')
    argparser.add_argument('--sweep_config', default='focused_sweep_config')
    argparser.add_argument('--model_args', default='bert_sweep_args')
    argparser.add_argument('--model_type', default='bert')
    argparser.add_argument('--model_name', default='bert-base-cased')
    args = argparser.parse_args()

    main(
        args.train_pkl,
        args.eval_pkl,
        args.config_json,
        args.sweep_config,
        args.model_args,
        args.model_type,
        args.model_name,
    )
