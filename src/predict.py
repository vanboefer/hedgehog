"""
Apply a fine-tuned token-level classification model to generate predictions.
The text is given in a pickled df and the predictions are generated per row and saved in a 'predictions' column.
"""


import argparse
import pandas as pd
from simpletransformers.ner import NERModel


def predict_df(
    data_pkl,
    model_type,
    model_name,
):
    """
    Apply a fine-tuned token-level classification model to generate predictions.
    The text is given in `data_pkl`; each row should be a sentence (sequence of tokens) and it is split by the model on spaces to generate predictions per token.
    The predictions for each row are saved in a 'predictions' column as a list.

    Parameters
    ----------
    data_pkl: str
        path to pickled df with the data, which must contain the column 'sentence'
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file

    Returns
    -------
    None
    """

    # load data
    df = pd.read_pickle(data_pkl)

    # load model
    model = NERModel(model_type, model_name)

    # predict
    def predict(sent):
        predictions, _ = model.predict([sent])
        return predictions

    df['predictions'] = df['sentence'].apply(predict)

    # pkl df
    df.to_pickle(data_pkl)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_pkl', default='../data/sentences.pkl')
    argparser.add_argument('--model_type', default='bert')
    argparser.add_argument('--model_name', default='models/hedgehog')
    args = argparser.parse_args()

    predict_df(
        args.data_pkl,
        args.model_type,
        args.model_name,
    )
