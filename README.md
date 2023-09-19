# Detecting Concepts and Generating Captions from Medical Images: Contributions of the VCMI Team to ImageCLEFmedical Caption 2023

This is the official repository for the [VCMI](https://vcmi.inesctec.pt)'s Team submission to [ImageCLEFmedical Caption 2023](https://www.imageclef.org/2023/medical/caption).

You can find the paper [here](#).

For more information please contact [isabel.riotorto@inesctec.pt](mailto:isabel.riotorto@inesctec.pt) or [maria.h.sampaio@inesctec.pt](mailto:maria.h.sampaio@inesctec.pt).



## Requirements
You can find the package requirements in the [requirements.txt](requirements.txt) file.



## Dataset Structure
```
dataset
    images/
    ImageCLEFmedical_Caption_2023_caption_prediction_train_labels.csv
    ImageCLEFmedical_Caption_2023_caption_prediction_valid_labels.csv
    ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv
    ImageCLEFmedical_Caption_2023_concept_detection_valid_labels.csv
```    



## Exploratory Data Analysis & Preprocessing
### Resize images (offline)
If you want to generate an offline resized version of the dataset, run:
```bash
$ python data_preprocessing/resize.py --original_path {path_to_dataset_dir} --new_path {path_to_resized_dataset_dir} --new_height {new_image_height}
```

This offline resized version of the dataset will speed-up the training procedures.


### Captions' Statistics
To get the statistics for the captions subset, run:
```bash
$ python data_preprocessing/captions_stats.py --base_dir {path_to_dataset_dir}
```


### Get all the concepts of the dataset (train & validation splits)
To get the list of all the concepts of the dataset (train & validation splits), run:
```bash
$ python data_preprocessing/get_all_concepts_from_csvs.py --base_dir {path_to_dataset_dir} --processed_dir {path_to_processed_dataset_dir}
```


### Get UMLS Information related to the concepts of the database
To get the UMLS Information related to the concepts of the database, run:
```bash
$ python data_preprocessing/get_umls_information.py --base_dir {path_to_dataset_dir} --processed_dir {path_to_processed_dataset_dir} --api_key {your_uts_api_key}
```

You can go to [https://documentation.uts.nlm.nih.gov/rest/authentication.html](https://documentation.uts.nlm.nih.gov/rest/authentication.html) for more information about getting your own API key.


### Merge Train and Validation Subsets
Although more useful for the competition, we leave you with a script to generate a single .CSV file with all the data samples (train and validation splits):
```bash
$ python data_preprocessing/merge_train_val.py --base_dir {path_to_dataset_dir} --base_file {path_to_the_ImageCLEFmedical_Caption_2023_concept_detection_train_labels.csv}
```



## Concept Detection
### Adversarial
To train the adversarial approach, run:
```bash
$ python concept_detection/adversarial/model_train.py {check_file_for_command_line_interface_arguments}
```

To generate predictions for the adversarial approach, run:
```bash
$ python concept_detection/adversarial/model_predict.py {check_file_for_command_line_interface_arguments}
```


### Evaluation
TBA


## Caption Prediction
TBA

### Evaluation
TBA
