# Dog Breed Classification Project

## Introduction
This project is a deep learning-based approach to classify dog breeds from images. It is inspired by Kaggle's Dog Breed Identification competition, which involves predicting the breed of a dog given its image. This task showcases the application of computer vision and transfer learning techniques.

## Problem Statement
The goal of this project is to identify the breed of a dog given an image. This involves building a model that can generalize well across different dog breeds.

## Dataset
The dataset is sourced from Kaggle's Dog Breed Identification competition. It contains:
- **Images**: High-quality images of dogs belonging to different breeds.
- **Labels**: A CSV file mapping each image to its corresponding breed.

## Techniques and Tools Used
- **Deep Learning Frameworks**: TensorFlow and TensorFlow Hub were used to build and train the model.
- **Transfer Learning**: Pre-trained models from TensorFlow Hub were leveraged to improve accuracy and reduce training time.
- **Data Visualization**: Tools like Matplotlib and Pandas were used to explore and understand the dataset.

## Data Preprocessing
1. **Loading and Visualizing Data**: Imported and explored the data using Pandas.
2. **Label Encoding**: Converted dog breeds into numerical labels.
3. **Image Resizing and Normalization**: Resized all images to a uniform shape and scaled pixel values to the range [0, 1].
4. **Augmentation**: Applied transformations such as flipping and rotation to increase dataset variability.

## Model Architecture
- Used a pre-trained **MobileNetV2** model from TensorFlow Hub as the base.
- Added custom dense layers on top of the base model for classification.
- Compiled the model using **Adam optimizer** and **categorical cross-entropy loss**.

## Training and Evaluation
- **Training**: The model was trained using an 80-20 train-validation split.
- **Evaluation Metrics**: Accuracy and loss were used to monitor performance.
- **Final Accuracy**: The model achieved a high accuracy on the validation set, indicating good generalization.

## Results
The model was able to classify dog breeds effectively, achieving:
- **Accuracy**: 69.0%
- **Loss**: 1.255

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dog-breed-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd dog-breed-classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook dog_vision.ipynb
   ```

## Dependencies
- **Libraries**:
  - TensorFlow
  - TensorFlow Hub
  - Pandas
  - Matplotlib
  - NumPy

## Conclusion
This project demonstrates the power of transfer learning in solving computer vision tasks like dog breed classification. By leveraging pre-trained models and fine-tuning them on a specific dataset, we achieved strong performance with relatively low computational resources.

---

### Author
Faraz Ashraf  

