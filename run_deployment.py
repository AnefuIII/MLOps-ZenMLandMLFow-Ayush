import click
from rich import print
from typing import cast

from pipelines.deployment_pipeline import (continuous_deployment_pipeline,
                                           inference_pipeline)

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
    )
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command()
@click.option(
    '--config',
    '-c',
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default = DEPLOY_AND_PREDICT,
    help = 'you can choose to only run the deployment pipeline'
    'to train and deploy a model ("DEPLOY"), or to only run a prediction '
    'against the deployed Model ("PREDICT")'
    'By default, both will be executed ("DEPLOY AND PREDICT").',

)

@click.option(
    '--min-accuracy',
    default = 0.0,
    help = 'Minimum accuracy to deploy the model',
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT

    if deploy:
        continuous_deployment_pipeline(
            data_path = "C:/Users/HP/Desktop/poetry/ayush/data/file1.csv",
            min_accuracy = min_accuracy,
            workers = 3,
            timeout = 60)
    if predict:
         inference_pipeline(
             pipeline_name = 'continuous_deployment_pipeline',
             pipeline_step_name = 'mlflow_model_deployer_step',
         )

    print(
        'you can run \n'
        f'[italic green] mlflow ui --backend-store-uri "{get_tracking_uri()}'
        '[/italic green] \n ... to start your experiment runs within MLFlow UI'
        '\n you can find your runs tracked within the '
        '"mlflow_example_pipeline". Then you will be able to'
        'compare 2 or more runs \n \n'
    )

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name= 'continuous_deployment_pipeline',
        pipeline_step_name= 'mlflow_model_deployer_step',
        model_name = 'model',
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])

        if service.is_running:
            print(
                f'The MLflow prediction server is running as a daemon'
                f'process service and accept inference requests at \n'
                f' {service.prediction_url} \n'
                f'To stop the service, run'
                f'[italic green] "zenml model-deployer models delete'
                f'{str(service.uuid)} [/italic green]'
            )
        elif service.is_failed:
            print(
                f'the Mlflow prediction server is in a failed state:\n'
                f'last state: "{service.status.state.value}" \n'
                f'last error: "{service.status.last_error}"'
            )
    else:
        print(
            'No Mlflow prediction server is currently running.'
            'The deployment pipeline must run first to train a '
            'model and deploy it. Execute the same command with '
            '"--deploy" argument to deploy a model.'
        )

if __name__ == '__main__':
    run_deployment()