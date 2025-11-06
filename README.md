## Project Overview

Understanding the emotions of infants is a challenging but important task. This project leverages deep learning to create a model capable of interpreting facial expressions from images of babies.

The solution involves:
* Processing and augmenting a specialized image dataset.
* Building a custom CNN inspired by **ResNet architecture** to combat the vanishing gradient problem.
* Training the model to classify images into one of five emotion categories.
* Evaluating the model's performance, which achieved a significant improvement over baseline models.

## Dataset

The model is trained on the **Baby Emotion Classification** dataset (available on Kaggle).

* **Source:** `../input/baby-emotion-classification/`
* **Classes:** The dataset contains images sorted into five categories:
    1.  Angry
    2.  Fear
    3.  Happy
    4.  Neutral
    5.  Sad
* **Structure:** The data is pre-split into `train` and `validation` sets.

## Methodology

### 1. Data Preprocessing & Augmentation

To create a more robust model and prevent overfitting, the training data was augmented using `ImageDataGenerator` from Keras. The following transformations were applied:
* **Rescaling:** Pixel values were normalized from `[0, 255]` to `[0, 1]`.
* **Horizontal Flips:** Images were randomly flipped horizontally.
* **Shift Augmentation:** Random width and height shifts (10%) were applied to make the model robust to variations in positioning.

### 2. Model Architecture: Custom ResNet

A standard, deep CNN can suffer from the vanishing gradient problem. To solve this, a custom model was built using **residual connections (or skip connections)**, which is the core concept of ResNet.

* The architecture consists of several convolutional blocks.
* Each block contains `Conv2D` layers, `BatchNormalization` (for stable training), and `ReLU` activation.
* A "skip connection" adds the input of a block to its output, allowing gradients to flow more easily through the network.
* The final layers consist of a `GlobalAveragePooling2D` layer, a `Dense` layer, and a `softmax` activation function to output probabilities for the five emotion classes.

## Results

The model was trained for **100 epochs** and achieved a **peak validation accuracy of approximately 85%**.

This 85% accuracy represents a strong improvement over simpler, "plain" CNN architectures, proving the effectiveness of the residual connections for this complex image classification task. The training and validation loss curves show good convergence, indicating the model learned effectively without significant overfitting.

### Training Performance

![Model Training History](images/training_history.png)

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
    cd [Your-Repo-Name]
    ```
2.  **Install dependencies:**
    The project primarily uses TensorFlow/Keras, NumPy, and Matplotlib.
    ```bash
    pip install tensorflow numpy matplotlib pandas
    ```
3.  **Get the data:**
    * Download the [Baby Emotion Classification](https://www.kaggle.com/datasets/sudarshanvaidya/baby-emotion-classification) dataset from Kaggle.
    * Place the `train` and `validation` folders in a directory structure that the notebook can access (e.g., in an `input/` folder).
4.  **Run the notebook:**
    * Open and run the `emotions-baby (1).ipynb` notebook using Jupyter or Google Colab.
