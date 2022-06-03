"""Extract and transform features."""

import os
import click
import warnings
import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import load, dump
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from sklearn.model_selection import train_test_split


def eda(labels, confounds=None, prefix: str = 'all'):
    """Explorative data analysis.

    Args:
        labels ([type]): Dependent variable of that we try to classify.
        confounds ([type], optional): Potencial confounds for classifier.
            Defaults to None.
        prefix (str, optional): Prefix of plot names. Defaults to 'all'.
    """

    sns.set(style='white',
            rc={'axes.facecolor': 'whitesmoke',
                'figure.facecolor': 'whitesmoke'})
    if confounds is not None:
        for i, f in enumerate(confounds.columns):
            fig = plt.figure(figsize=(15, 8))
            ax = sns.kdeplot(data=confounds, x=f, fill=True, alpha=1)
            ax.set_title(f'confounds {prefix}: {f}')
            name_plt = f'{prefix}_confound_{f}.png'
            fig.savefig(name_plt)
            mlflow.log_artifact(name_plt)

    # labels
    fig = plt.figure(figsize=(15, 8))
    ax = sns.histplot(data=labels, discrete=True)
    ax.set_title(f'Dependent variable {prefix}', size=20, weight='bold')
    fig.savefig(f'{prefix}_labels.png')
    mlflow.log_artifact(f'{prefix}_labels.png')


@click.command()
@click.option("--data_source", type=str, help="Path to read asd data")
def etl(data_source: str,
        test_split: float = 0.3,
        seed: int = 42,
        decomposition: str = 'atlas',
        connectivity: str = 'partial correlation',
        name_confounds: list = ['SEX', 'AGE_AT_SCAN', 'HANDEDNESS_SCORES']):
    """Extracts, loads and transforms previous data dump.

    Args:
        data_source (str): path of data dump.
        test_split (float, optional): Ratio of test set. Defaults to 0.3.
        seed (int, optional): Seed. Defaults to 42.
        decomposition (str, optional): Source of rois. Defaults to 'atlas'.
        connectivity (str, optional): Common connectivity estimator. 
            Defaults to 'partial correlation'.
        name_confounds (list, optional): Variable name of potencial confounds,
            extracted from phenotypes. Defaults to ['SEX', 'AGE_AT_SCAN',
            'HANDEDNESS_SCORES'].
    """
    warnings.filterwarnings("ignore")
    with mlflow.start_run() as mlrun:
        mlflow.set_tag('mlflow.user', os.getenv('MLFLOW_TRACKING_USERNAME'))

        # load
        feature_filename = os.path.join(data_source,
                                        f'roi_{decomposition}.dump')
        with open(feature_filename, 'rb') as filehandle:
            raw = load(filehandle)
        phenotypes = pd.read_csv(os.path.join(data_source, 'phenotypes.csv'))

        # extract
        labels = phenotypes.DX_GROUP
        confounds = phenotypes[name_confounds]

        # transform
        conn_est = ConnectivityMeasure(kind=connectivity)
        features = conn_est.fit_transform(raw)
        features = sym_matrix_to_vec(features)

        (x_train,
         x_test,
         y_train,
         y_test) = train_test_split(features,
                                    labels,
                                    test_size=test_split,
                                    stratify=labels,
                                    random_state=seed)

        eda(labels, confounds)

        # store results
        y_train = y_train.rename('labels')
        y_train.to_csv('labels_train.csv')
        mlflow.log_artifact('labels_train.csv')
        y_test = y_test.rename('labels')
        y_test.to_csv('labels_test.csv')
        mlflow.log_artifact('labels_test.csv')

        confounds_train = confounds.iloc[y_train.index.values, :]
        confounds_train.to_csv('confounds_train.csv')
        mlflow.log_artifact('confounds_train.csv')

        with open('features_train.dump', 'wb') as filehandle:
            dump(x_train, filehandle, compress=('lzma', 9))
        mlflow.log_artifact('features_train.dump')

        with open('features_test.dump', 'wb') as filehandle:
            dump(x_test, filehandle, compress=('lzma', 9))
        mlflow.log_artifact('features_test.dump')


if __name__ == "__main__":
    etl()
