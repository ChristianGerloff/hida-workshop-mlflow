"""Example to show how models can be loaded."""

import os
import click
import mlflow
import pandas as pd

from joblib import load


@click.command()
@click.option("--x_source", type=str, help="Path to features")
@click.option("--y_source", type=str, help="Path to labels")
@click.option("--model_source", type=str, help="Path to model")
def test_model(x_source: str,
               y_source: str,
               model_source: str):

    # load test data
    with open(x_source, 'rb') as filehandle:
        x_data = load(filehandle)
    y_data = pd.read_csv(y_source, index_col=0)
    y_data = y_data['labels']

    model = mlflow.sklearn.load_model(model_source)

    with mlflow.start_run() as run:
        mlflow.set_tag('mlflow.user', os.getenv('MLFLOW_TRACKING_USERNAME'))

        # results of complete training data
        mlflow.sklearn.eval_and_log_metrics(model,
                                            x_data,
                                            y_data,
                                            prefix="test_")


if __name__ == "__main__":
    test_model()
