# LAB-Deep-Learning-Neural-Networks-and-Feedforward-Neural-Networks

![](https://user-images.githubusercontent.com/970858/63474771-d6734700-c469-11e9-83bb-9429da563909.png)

## Introduction

In this LAB, we will walk through a comprehensive Deep Learning workflow using both the **Bike Sharing dataset** and the **Pima Indians Diabetes dataset**. The **Bike Sharing dataset** contains daily records of bike rentals in a specific city, while the **Pima Indians Diabetes dataset** is focused on medical records used to predict the onset of diabetes. Both datasets will allow us to explore a variety of deep learning tasks, from regression to binary classification.

The goal is to build and compare Feedforward Neural Networks (FNNs) to:

- **Iteration 1**: Predict the total number of bike rentals (`cnt`) using the Bike Sharing dataset.
- **Iteration 2**: Predict the onset of diabetes (binary classification) using the Pima Indians Diabetes dataset.

We will explore various architectures of FNNs, including different numbers of layers, units, and activation functions for each dataset. Additionally, we will implement regularization techniques such as dropout and L2 regularization to improve model generalization and prevent overfitting.

To optimize the performance of our neural networks, we will use **RandomizedSearchCV** for hyperparameter tuning, which will help in selecting the best parameters and configurations for both the regression and classification models.

---

## Setup Instructions

To run the notebook and reproduce the results, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/codeskills-ai/LAB-Deep-Learning-Neural-Networks-and-Feedforward-Neural-Networks.git
    cd LAB-Deep-Learning-Neural-Networks-and-Feedforward-Neural-Networks
    ```

2. **Create a virtual environment and install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

---

## Iteration 1 - Bike Sharing 🚲

In this iteration, you will train a neural network on the `Bike Sharing` dataset. The goal is to predict the total number of daily bike rentals (cnt) based on various features such as weather conditions, season, and holiday information. You will explore how different neural network architectures impact regression performance and evaluate model generalization on unseen data. This will help you understand how to build and optimize Feedforward Neural Networks for regression tasks.

### Step 1: Data Load

Load the Bike Sharing Dataset into a pandas DataFrame. Follow the same steps as outlined in the regression lab for downloading and loading the dataset.

- Import necessary libraries: `pandas`, `zipfile`, `io`, and `requests`.
- Download and unzip the dataset.
- Load the `day.csv` file into a pandas DataFrame.
- Display the first few rows to understand its structure.

### Step 2: Explore Data

Visualize the dataset to understand the distribution of `cnt` and its relationship with other features.

- Import visualization libraries: `seaborn` and `matplotlib.pyplot`.
- Visualize the distribution of `cnt` and scatter plots of features against `cnt`.

### Step 3: Feature Selection and Preprocessing

#### 3.0 Remove Unnecessary Columns

Remove columns that are not needed for the model, such as `dteday`.

#### 3.1 Select X (Features) and y (Target)

Select the feature set `X` (all columns except `cnt`) and the target variable `y` (the `cnt` column). Then, split the data into training and testing sets:

#### 3.2 Split Data into Training and Testing Sets

Split the data into 80% training and 20% testing sets:

#### 3.3 Normalize Features

Normalize the feature set to ensure that each feature contributes equally to the training process.

### Step 4: Build Two Feedforward Neural Networks

In this step, you will create and compare **two feedforward neural network models** to predict the total number of bike rentals (`cnt`). One model will be created without any regularization techniques, while the other will include both **dropout** and **L2 regularization**. This will help you understand the impact of these techniques on model performance.

#### 4.1: Build a Feedforward Neural Network without Regularization or Dropout

1. **Define the Model Architecture**

   - **Input layer**: Define the input shape using the number of features in the dataset (`X_train.shape[1]`).
   - **First hidden layer**: Add a dense layer with 32 neurons and the ReLU activation function.
   - **Second hidden layer**: Add a dense layer with 12 neurons and the ReLU activation function.
   -**Output layer**: Add a dense layer with 1 neuron and no activation function for regression.


2. **Compile the model**:
   - Use an appropriate optimizer (e.g., Adam).
   - Set the loss function to Mean Squared Error (MSE).
   - Include Mean Absolute Error (MAE) as a metric.

#### 4.2: Build a Feedforward Neural Network with Dropout and L2 Regularization

1. **Define the Model Architecture**

   - **Input layer**: Specify the input shape based on the number of features in the dataset (`X_train.shape[1]`).
   - **First hidden layer**: Add a dense layer with 32 neurons, the ReLU activation function, and set the input shape.
   - **Dropout layer**: Add a dropout layer with a rate of 0.1 after the first hidden layer.
   - **Second hidden layer**: Add a dense layer with 12 neurons and the ReLU activation function.
   - **Dropout layer**: Add another dropout layer with a rate of 0.1 after the second hidden layer.
   - **Output layer**: Add a dense layer with 1 neuron and no activation function for regression.


2. **Compile the model**:
   - Use an appropriate optimizer (e.g., Adam).
   - Set the loss function to Mean Squared Error (MSE).
   - Include Mean Absolute Error (MAE) as a metric.

---

### Step 5: Train Both Neural Networks

Train **both neural network models** on the training data:

1. Train the model without regularization or dropout using the following parameters:
   - Number of epochs: 100
   - Batch size: 32

2. Train the model with dropout and L2 regularization using the same parameters:
   - Number of epochs: 100
   - Batch size: 32

---

### Step 6: Evaluate Both Models

Evaluate the performance of each model on the test data:

1. For the model without regularization or dropout, calculate:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)

2. For the model with dropout and L2 regularization, calculate:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)

Compare the results to see how regularization techniques affect the performance.

---

### Step 7: Visualize Training History

Plot and compare the training and validation loss for both models over epochs:

1. Create a plot for the model without regularization or dropout showing:
   - Training Loss
   - Validation Loss

2. Create a plot for the model with dropout and L2 regularization showing:
   - Training Loss
   - Validation Loss

Analyze the plots to diagnose the training process and observe differences.

---

### Step 8: Hyperparameter Tuning

Optimize both neural networks by experimenting with different hyperparameters:

1. Tune the number of layers, units per layer, and dropout rates.
2. Use techniques such as **RandomSearchedCV** from `sklearn`, or perform manual tuning.

---

### Step 9: Final Model Evaluation

After hyperparameter tuning, evaluate the final models:

1. Print and compare the final Mean Squared Error (MSE) and Mean Absolute Error (MAE) for both models.
2. Plot and compare the training and validation loss over epochs for both final models.

---

## Iteration 2 - Pima Indians Diabetes Dataset 🏥

In this iteration, you will train a neural network on the **Pima Indians Diabetes dataset**. The objective is to observe overfitting and understand how it affects model performance. Overfitting occurs when a model performs well on training data but fails to generalize to unseen data.

### Step 1: Data Load

Load the **Pima Indians Diabetes dataset** into a pandas DataFrame from the URL provided. Display the first few rows to understand its structure.

### Step 2: Explore Data

Visualize the dataset to understand the distribution of the features and the outcome variable (`Outcome`).

### Step 3: Define Features and Labels

Separate the features (`X`) from the labels (`y`).

- **Features (`X`)**: Include all columns except `'Outcome'`.
- **Labels (`y`)**: Contain only the `'Outcome'` column.

### Step 4: Split Data into Training and Validation Sets

Split the dataset into training and validation sets (80% training, 20% validation).

### Step 5: Normalize Features

Normalize the features to ensure each feature contributes equally to the training process.

### Step 6: Build a Feedforward Neural Network

Create a feedforward neural network to predict the onset of diabetes.

### Step 7: Train the Model

Train the model on the training data and validate on the validation set.

### Step 8: Evaluate the Model

Evaluate the model's performance on the validation data.

### Step 9: Visualize Training History

Plot the training and validation accuracy over epochs.

---
## Bonus

### Compare Results with Other Models

Compare the performance of the feedforward neural network with Linear Regression, Polynomial Regression, Ridge, and Lasso Regression models.

- Compare metrics such as MSE and R² to determine which model performs best.

## Submission
Once you have completed your analysis and visualizations, push your Jupyter notebook to your forked GitHub repository. Ensure that your final submission includes narratives explaining your visualizations and insights.


```bash
$ git add .
$ git commit -m "done"
$ git push origin main
```

Then create a Pull Request!!

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API](https://keras.io/)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
- [Scikit-learn RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)


![happy_coding](https://user-images.githubusercontent.com/970858/63899010-c23fc480-c9ea-11e9-84a2-542907e42362.png)
