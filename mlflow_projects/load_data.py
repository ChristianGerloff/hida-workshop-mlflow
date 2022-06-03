"""
Downloads the imaging and phenotype data and stores them as artifacts.
"""

import os
import warnings
import click
import boto3
import mlflow


def download_s3(client, resource, bucket: str, dist: str, local='/tmp'):
    """Downloads files from subfolders in s3.

    Args:
        client (boto.client): client of aws resource.
        resource (boto.resource): resource in aws.
        bucket (str): Name of source bucket.
        dist (str): subfolder.
        local (str, optional): [description]. Defaults to '/tmp'.
    """
    paginator = client.get_paginator('list_objects')
    pag = paginator.paginate(
        Bucket=bucket, Delimiter='/', Prefix=dist)
    for result in pag:
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_s3(client,
                            resource,
                            bucket,
                            subdir.get('Prefix'),
                            local)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if file.get('Key')[-1] != "/":
                resource.meta.client.download_file(
                    bucket, file.get('Key'), dest_pathname)


@click.command()
@click.option("--bucket", type=str, default='hida-workshop-data')
@click.option("--dataset", type=str, default='hdslee')
def load_data(bucket: str = 'hida-workshop-data', dataset: str = 'hdslee'):
    """Loads data from aws buckets.

    Args:
        bucket (str, optional): Name of source bucket. Defaults to 'hida-workshop-data'.
        dataset (str, optional): Name of dataset that will be downloaded.
             Defaults to 'hdslee'.
    """
    warnings.filterwarnings("ignore")
    key_id = os.getenv('AWS_ACCESS_KEY_ID')
    key = os.getenv('AWS_SECRET_ACCESS_KEY')

    with mlflow.start_run() as mlrun:
        mlflow.set_tag('mlflow.user', os.getenv('MLFLOW_TRACKING_USERNAME'))

        client = boto3.client('s3',
                              aws_access_key_id=key_id,
                              aws_secret_access_key=key)
        s3 = boto3.resource('s3')

        download_s3(client,
                    s3,
                    bucket,
                    dataset,
                    '/data')

        mlflow.log_artifacts('/data/'+dataset)


if __name__ == "__main__":
    load_data()
