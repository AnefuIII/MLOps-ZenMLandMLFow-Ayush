
from pipelines.training_pipeline import train_pipeline 
from zenml.client import Client

if __name__ == "__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = "C:/Users/HP/Desktop/poetry/ayush/data/file1.csv")

#tracking uri interface
#run in terminal:
# mlflow ui --backend-store-uri 'file location to uri'
   