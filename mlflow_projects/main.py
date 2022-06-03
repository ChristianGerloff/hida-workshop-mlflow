"""
Starting script of dependent multi stage pipeline.
"""

import os
import mlflow
import click

from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id


def _already_ran(entry_point_name: str,
                 parameters: dict,
                 git_commit: str,
                 user_name: str = None,
                 experiment_id: str = None) -> str:
    """Detect if a run with the given entrypoint name,
       parameters, experiment id from a specific user already ran.

    Args:
        entry_point_name (str): Name of pipeline stage.
        parameters (dict): Parameters of pipeline stage.
        git_commit (str): Id of corresponding git commit.
        user_name (str, optional): Filter history by this name.
            Defaults to None.
        experiment_id (str, optional): Id of relevant experiment.
            Defaults to None.

    Returns:
        str: id of identical historical run.
    """
    experiment_id = (experiment_id if experiment_id is not None
                     else _get_experiment_id())
    client = mlflow.tracking.MlflowClient()

    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        hist_entry_point = tags.get(
            mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None)
        if hist_entry_point != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue
        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(f'Run matched, but is not FINISHED, so skipping'
                   f'run_id={run_info.run_id}, status= {run_info.status}')
            continue
        hist_user_name = tags.get(mlflow_tags.MLFLOW_USER, None)
        if (user_name is not None and hist_user_name != user_name):
            eprint(f'Run matched, but user does not match'
                   f'found={hist_user_name}, '
                   f'current user={user_name}')
            continue
        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(f'Run matched, but has a different source version, '
                   f'so skipping found={previous_version}, '
                   f'expected={git_commit}')
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None


def _get_or_run(entrypoint: str,
                parameters: dict,
                git_commit: str,
                user_name: str = None,
                use_cache: bool = True):
    """Provides information about current run of pipeline stage.

    Args:
        entrypoint (str): Name of pipeline stage.
        parameters (dict): Parameters of pipeline entry point.
        git_commit (str): Corresponding git commit.
        user_name (str, optional): User name. Defaults to None.
        use_cache (bool, optional): Check history to avoid duplicate runs.
            Defaults to True.

    Returns:
        [type]: [description]
    """
    if user_name is not None:
        existing_run = _already_ran(entrypoint,
                                    parameters,
                                    git_commit,
                                    user_name)
    else:
        existing_run = _already_ran(entrypoint, parameters, git_commit)
    if use_cache and existing_run:
        print(f'Found existing run for entrypoint={entrypoint} '
              f'and parameters={parameters}')
        return existing_run
    print(f'Launching new run for entrypoint={entrypoint} '
          f'and parameters={parameters}')
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)

    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option('--bucket', default=os.getenv('BUCKET_NAME'), type=str)
@click.option('--dataset')
@click.option('--alpha')
@click.option('--n_components')
def workflow(bucket: str,
             dataset: str,
             alpha: float = 0.9,
             n_components: int = 10):

    user_name = os.getenv('MLFLOW_TRACKING_USERNAME')
    with mlflow.start_run() as active_run:
        mlflow.set_tag('mlflow.user', user_name)
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        load_data_run = _get_or_run('load_data',
                                    {'bucket': bucket, 'dataset': dataset},
                                    git_commit,
                                    user_name)
        artifact_uri_ldr = load_data_run.info.artifact_uri
        etl_run = _get_or_run('etl',
                              {'data_source': artifact_uri_ldr},
                              git_commit,
                              user_name)
        artifact_uri_etl = etl_run.info.artifact_uri
        x_source = os.path.join(artifact_uri_etl, 'features_train.dump')
        y_source = os.path.join(artifact_uri_etl, 'labels_train.csv')
        sklearn_cls_run = _get_or_run('sklearn_cls',
                                      {'x_source': x_source,
                                       'y_source': y_source,
                                       'alpha': alpha,
                                       'n_components': n_components},
                                      git_commit,
                                      user_name)
        artifact_uri_cls = os.path.join(sklearn_cls_run.info.artifact_uri,
                                        'model')
        x_source = os.path.join(artifact_uri_etl, 'features_test.dump')
        y_source = os.path.join(artifact_uri_etl, 'labels_test.csv')
        sklearn_test_run = _get_or_run('sklearn_test',
                                       {'x_source': x_source,
                                        'y_source': y_source,
                                        'model_source': artifact_uri_cls},
                                       git_commit,
                                       user_name)


if __name__ == "__main__":
    workflow()
