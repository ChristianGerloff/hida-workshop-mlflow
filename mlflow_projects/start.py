import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URL'))
mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT'))

mlflow.projects.run('.',  # git URL
                    backend='local',
                    synchronous=True,
                    entry_point='main',
                    parameters={'bucket': os.getenv('BUCKET_NAME'),
                                'dataset': 'asd',
                                'alpha': 0.9,
                                'n_components': 20})

mlflow.projects.run('.',
                    backend='local',
                    synchronous=True,
                    entry_point='main',
                    parameters={'bucket': os.getenv('BUCKET_NAME'),
                                'dataset': 'asd',
                                'alpha': 0.9,
                                'n_components': 5})
