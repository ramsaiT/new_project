"""
=========================  Please Read this before Coding  =========================
Given below is the pipeline function is where you are going to add your custom code
We have already imported libraries like ProjectDataManager, ModelRegHelper

---- Reading and Saving Data ----
ProjectDataManager is used to read and save data from the input, output and model folders.
An data_manager_object is already created for you so that you can read and write files from your project

It has various functions whose uses are given below:
1. read_file_from_data_inputs: function to read data from Data Inputs folder using pandas, it requires the file name
2. read_file_from_data_outputs: function to read data from Data Outputs folder using pandas, it requires the file name
3. read_model_from_model_folder: function to read a model from Model folder using joblib, it requires the file name
4. save_file_to_data_inputs: saves a data frame to the Data Inputs Folder, requires data frame and file name
5. save_file_to_data_outputs: saves a data frame to the Data Outputs Folder, requires data frame and file name

---- Saving Models ----
If you are using this pipeline to Train models then you can use the ModelRegHelper class to upload the model
to our Model Registry.
If you want to save the model you will have to create an object of this class and pass it certain parameters based on
the model type.

For Segmentation Models like k_means you will need to pass:
--> trained_model, model_type, train_data, columns_used_to_train, model_name, auto, input_path, output_path, model_path.
--> And you will call the "save_clustering_model" function.

For Regression and Classification Models you will need to pass:
--> trained_model, model_type, train_data, columns_used_to_train, test_data, target_column, model_name, auto,
    input_path, output_path, model_path.
--> And you will call the "save_classification_or_regression_model" function.

Explanation of the Arguments used for the Model Registry:
trained_model: the actual model object
1. model_type: type of the model i.e. if its a clustering model then it will be clustering, for Regression it will be
regression and for Segmentation it will be classification.
2. train_data: Data Used to Train the Model with the target column in it.
3. columns_used_to_train: Columns Used to train the model.
4. test_data: Test data used for validating the model  with the target column in it, will be None for Segmentation models.
5. target_column: Column name of the target column, will be None if the model os a CLUSTERING type.
6. model_name: Name of the model file, can be None and in that case
7. auto: This value should always be True
8. input_path: The input_path variable that has the path to the Data Inputs Folder
9. output_path: The output_path variable that has the path to the Data Outputs Folder
10. model_path: The model_path variable that has the path to the Model Folder

Example to save a model:

--> Segmentation model:
model_saver_object = ModelRegHelper(trained_model=trained_model_object,
									model_type="clustering", train_data=X_train_data_frame,
									columns_used_to_train=column_list_used_to_train, test_data=None,
									target_column=None, model_name="test_model_name", auto=True,
									input_path=input_path, output_path=output_path, model_path=model_path)
model_saver_object.save_clustering_model()

--> Regression or Classification model
model_saver_object = ModelRegHelper(trained_model=trained_model_object,
									model_type="regression" or "classification", train_data=train_data_frame,
									columns_used_to_train=column_list_used_to_train, test_data=test_data_frame,
									target_column=target_column_name, model_name="test_model_name", auto=True,
									input_path=input_path, output_path=output_path, model_path=model_path)
model_saver_object.save_classification_or_regression_model()

************* Happy Coding From Fluid AI *************
"""

import os
import pandas as pd
import datetime

from fluidai_net.project_manager.project_data_manager import ProjectDataManager
from fluidai_net.project_manager.modelreg_helper import ModelRegHelper


def pipeline(master_conf, cleaned_data, output_name, output_format):
    """
    This pipeline function is used to execute the code within it using a yaml file from the dashboard
    :param master_conf: a dictionary that has all the paths for the various folders in a Particular Project
    :param output_name: Name of the Output File
    :param output_format: Format in which the file needs to be saved
    All other pipeline function arguments above are the names of the files that are to be used by this function.
    :return:
    """
    # Loads the folder paths for the project version
    input_path = master_conf["input_path"]
    output_path = master_conf["output_path"]
    schema_path = master_conf["schema_path"]
    model_path = master_conf["model_path"]
    vec_meta_path = master_conf["vec_meta_path"]

    data_manager_object = ProjectDataManager(master_conf)

    # examples to read data from input files
    cleaned_data = pd.read_csv(os.path.join(input_path, cleaned_data))

    # get current date time

    current_datetime = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S').replace("-", "_").replace(" ",
                                                                                                       "-").replace(
        ":", "_")

	
    # examples to save data into Data Inputs
    cleaned_data = cleaned_data.iloc[:100]
    cleaned_data.to_csv(os.path.join(output_path, output_name + output_format), index=False)
    cleaned_data.to_csv(os.path.join(output_path, output_name + current_datetime + output_format), index=False)
