"""
Evaluate a fine-tuned token-level classification model on a test set.
Save the following outputs:
- the `eval_results.txt` file in the model directory, contains the metrics:
    - classification report
    - confusion matrix
    - eval loss
    - macro-averaged precision
    - macro-averaged recall
    - macro-averaged F1-score
- model predictions, saved in the path indicated by the '--output' param
"""


import argparse
import pickle
import pandas as pd
from simpletransformers.ner import NERModel
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from itertools import chain


def evaluate(
    test_pkl,
    model_type,
    model_name,
    output_path,
):
    """
    Evaluate a fine-tuned token-level classification model on a test set.
    Save evaluation metrics in a `eval_results.txt` file in the model directory. The metrics include: classification report, confusion matrix, eval loss, macro-averaged precision, macro-averaged recall, macro-averaged F1-score.
    Save model predictions in a pickled file at `output_path`.

    Parameters
    ----------
    test_pkl: str
        path to pickled df with the test data, which must contain the columns 'sentence_id', 'words' and 'labels'
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file
    output_path: str
        path to save the pickled model predictions

    Returns
    -------
    None
    """

    # load data
    test_data = pd.read_pickle(test_pkl)

    # load model
    model = NERModel(model_type, model_name)

    # evaluate model
    flatten = lambda lst: list(chain(*lst))
    labels = ['C', 'D', 'E', 'I', 'N']
    results, _, predictions = model.eval_model(
        test_data,
        precision   =lambda y_true, y_pred: precision_score(flatten(y_true), flatten(y_pred), average='macro'),
        recall      =lambda y_true, y_pred: recall_score(flatten(y_true), flatten(y_pred), average='macro'),
        f1_score    =lambda y_true, y_pred: f1_score(flatten(y_true), flatten(y_pred), average='macro'),
        class_report=lambda y_true, y_pred: classification_report(flatten(y_true), flatten(y_pred)),
        conf_matrix =lambda y_true, y_pred: confusion_matrix(flatten(y_true), flatten(y_pred), labels=labels),
    )

    print(results['class_report'])

    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test_pkl', default='../data/test.pkl')
    argparser.add_argument('--model_type', default='bert')
    argparser.add_argument('--model_name', default='models/hedgehog')
    argparser.add_argument('--output', default='models/hedgehog/predictions.pkl')
    args = argparser.parse_args()

    evaluate(
        args.test_pkl,
        args.model_type,
        args.model_name,
        args.output,
    )
