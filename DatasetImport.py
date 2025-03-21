import kagglehub
import os

# SnP dataset
def getSnPDatasetPath():
    path_to_s_and_p_dataset = kagglehub.dataset_download("yash16jr/s-and-p500-daily-update-dataset", path=os.getcwd())
    print("Path to dataset files:", path_to_s_and_p_dataset)
    return path_to_s_and_p_dataset

# Car dataset
def getCarDatasetPath():
    path_to_car_dataset = ".venv/include/car_price_dataset.csv"
    print("Path to dataset files:", path_to_car_dataset)
    return path_to_car_dataset

getCarDatasetPath()