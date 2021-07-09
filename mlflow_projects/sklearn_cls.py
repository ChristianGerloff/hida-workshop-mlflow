"""
Example of a simple calssifier for within sample classification which
can be used further for prediction on the test set.
"""
import os
import click
import mlflow
import pandas as pd

from joblib import load
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from mlflow.models.signature import infer_signature
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

@click.command()
@click.option("--x_source", type=str, help="Path to features")
@click.option("--y_source", type=str, help="Path to labels")
@click.option("--alpha", type=float, default=0.9)
@click.option("--n_components", type=int, default=10)
def classify(x_source:str,
             y_source:str,
             alpha:float=0.9,
             n_components:int=10,
             k_folds:int=5,
             scoring:str='roc_auc',
             seed:int=42):

    NOTE = ('Metrics with the prefix `train_all` correspond ' 
            'to training on all data, while `training`  '
            'correspond to the cross-validation results. \n '
            '***Attention*** this means that the plots '
            'of the training correspond only to the last fold of the CV!')

    # load
    with open(x_source, 'rb') as filehandle:
        x_data = load(filehandle)
    y_data = pd.read_csv(y_source, index_col=0)
    y_data = y_data['labels']

    # define pipeline
    pipe = Pipeline(
        steps=[('decomp', TruncatedSVD(n_components=n_components, random_state=seed)),
               ('cls', RidgeClassifier(alpha=alpha, random_state=seed))])
    pipe.fit(x_data, y_data)
    signature = infer_signature(x_data, pipe.predict(x_data))

    # start tracking
    mlflow.sklearn.autolog(log_model_signatures=False, log_models=False)
    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.note.content', NOTE)
        mlflow.set_tag('mlflow.user', os.getenv('MLFLOW_TRACKING_USERNAME'))

        # results of complete training data
        mlflow.sklearn.log_model(pipe, "model", signature=signature)
        mlflow.sklearn.eval_and_log_metrics(pipe,
                                            x_data,
                                            y_data,
                                            prefix="train_all_")
        
        scores = cross_val_score(
            pipe, x_data, y_data, cv=k_folds, scoring=scoring)

        mlflow.log_metrics({f'{scoring}_mean': scores.mean()})
        for i, s in enumerate(scores):
            mlflow.log_metrics({scoring: s}, step=i+1)

        
if __name__ == "__main__":
    classify()
