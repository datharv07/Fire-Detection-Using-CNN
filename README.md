# Fire Detection

This repository contains code for a Fire Detection system using deep learning techniques. The system utilizes the ResNet50 architecture pre-trained on the ImageNet dataset and is fine-tuned for the task of detecting fires in images.

## Overview

The project consists of the following main components:

1. Data Collection: The dataset used for training, validation, and testing is organized into directories containing images of fire and non-fire scenarios.

2. Data Preprocessing: Data augmentation techniques are applied to generate additional training samples and enhance the diversity of the dataset.

3. Model Training: A deep learning model based on the ResNet50 architecture is trained using the preprocessed data. The training script can be found in the `training.py` file.

4. Model Evaluation: The trained model is evaluated on a separate test dataset to assess its performance in detecting fires.

5. Prediction: The trained model can be used to predict whether a given image contains a fire or not. An example of how to use the model is provided in the `Model_Use.ipynb` notebook.

6. Deployment: The deployment folder contains a `Main.py` file which serves as the entry point for deploying the model locally. To deploy the model, run the following command:

```bash
streamlit run deployment/Main.py
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/datharv07/Fire-Detection.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python training.ipynb
```


4. Make predictions:

```bash
python Model Use .ipynb path_to_image
```

Replace `path_to_image` with the path to the image you want to make predictions on.

## Screenshots
![image](https://github.com/datharv07/Fire-Detection-Using-CNN/assets/113291891/06163a69-f476-4c7e-9833-a74e194766e7)


## Requirements

The following Python libraries are required to run the code:

- tensorflow
- keras
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- opencv-python
- tqdm
- pillow
- scikit-image

These dependencies can be installed using the `requirements.txt` file provided in the repository.

## Contributors

- [Daware Atharv](https://github.com/datharv07)
- [Kaggel Data Set ](https://www.kaggle.com/datasets/atulyakumar98/test-dataset/code)
- 

