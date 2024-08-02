# Landmark Classification & Tagging for Social Media



## Project Overview
Welcome to the "Landmark Classification & Tagging for Social Media" project! This project focuses on building a landmark classifier.

## Project Steps

The project is divided into the following key steps:

1. ** Develop a CNN for Landmark Classification (from Scratch)** This step involves visualizing the dataset, processing it for training, and constructing a CNN from the ground up to classify landmarks. I will outline the data processing choices and network architecture. The best network model will be exported using Torch Script.

2. **Utilize Transfer Learning for Landmark Classification** In this step, I will explore pre-trained models, select one, and fine-tune it for landmark classification. The choice of the pre-trained network will be explained, and the final model will be exported using Torch Script.

3. **Deploy the Model in an App** The top-performing model will be used to create a user-friendly app that predicts likely landmarks in images. The model will be tested, and its strengths and weaknesses will be assessed.

The project starter kit includes the following notebooks:

- [cnn_from_scratch.ipynb](cnn_from_scratch.ipynb): Develop a CNN from scratch.
- [transfer_learning.ipynb](transfer_learning.ipynb): Apply transfer learning.
- [app.ipynb](app.ipynb): Deploy the best model in an app. Generate the archive file for submission.

## Project Purpose

Photo-sharing and storage services greatly benefit from attaching location data to uploaded photos. However, many photos lack this metadata, making it difficult to enhance user experiences. This project aims to address this problem by automatically predicting image locations through landmark classification.

When location metadata is missing, inferring the location from recognizable landmarks becomes a viable solution. Given the sheer volume of images uploaded to such services, manual landmark classification isn't feasible. This project takes the initial steps toward solving this issue by building models to predict image locations based on the landmarks they depict.




## Dataset and Models

### Dataset

The landmark images used in this project are a subset of the Google Landmarks Dataset v2.

### Models and Accuracy

Two approaches were explored for classifying landmarks:

#### CNN from Scratch

I designed a custom Convolutional Neural Network (CNN) architecture and trained from scratch to classify landmarks. This model was tailored to the specific requirements of the project and underwent rigorous training. It achieved 53% Accuracy.A custom Convolutional Neural Network (CNN) was designed and trained from scratch to classify landmarks. This model was specifically tailored for the project and underwent extensive training, achieving a 53% accuracy.

- **Model Architecture:** The architecture consists of 5 convolutional layers, providing the model with sufficient expressiveness. Dropout layers were included to reduce overfitting, and the model was designed to output a 50-dimensional vector corresponding to the 50 landmark classes.
- **Data Preprocessing:** Images were resized to 256 and then cropped to 224, the recommended input size for PyTorch's pre-trained models. Data augmentation was applied using RandAugment to enhance the model's robustness and improve test accuracy.
- **Training and Validation:** The model was trained for 50 epochs using the Adam optimizer and a learning rate scheduler. The weights corresponding to the lowest loss were saved.
- **Accuracy:** 53%

#### Transfer Learning

Transfer learning involves using pre-trained CNN models and fine-tuning them for the landmark classification task. This approach leverages knowledge from a large dataset and adapts it to the specific task.

- **Pre-trained Model Selection:**  ResNet50 was chosen as the base model due to its depth and popularity. It was fine-tuned for the landmark classification task.
- **Training and Validation:** The same process as with the CNN from scratch was followed.
- **Accuracy:** 74%

### Performance Evaluation

Both models were thoroughly evaluated and compared to assess their effectiveness in accurately classifying landmarks. The transfer learning model was ultimately selected for deployment due to its superior performance and ability to generalize to new, unseen images.


By Bhagya Sri Uddandam
