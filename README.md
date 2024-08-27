# LAB-Deep-Learning-Neural-Networks-and-Feedforward-Neural-Networks

![](https://user-images.githubusercontent.com/970858/63474771-d6734700-c469-11e9-83bb-9429da563909.png)


# Bike Sharing Dataset Analysis

## Introduction

In this LAB, we will walk through a comprehensive Deep Learning workflow using the Bike Sharing dataset. The dataset contains daily records of bike rentals in a specific city. We will cover data collection, preprocessing, feature engineering, model training, evaluation, and comparison of different deep learning techniques.

The goal is to build and compare Feedforward Neural Networks (FNNs) to predict the total number of bike rentals (cnt). We will explore various architectures of FNNs, including different numbers of layers and units, as well as activation functions. Additionally, we will implement regularization techniques such as dropout and L2 regularization to improve model generalization.

To optimize the performance of our neural networks, we will use GridSearchCV for hyperparameter tuning, which will help in selecting the best parameters and configurations for our models.

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

### Iteration 1: Data Load

Load the Bike Sharing Dataset into a pandas DataFrame. Follow the same steps as outlined in the regression lab for downloading and loading the dataset.

- Import necessary libraries: `pandas`, `zipfile`, `io`, and `requests`.
- Download and unzip the dataset.
- Load the `day.csv` file into a pandas DataFrame.
- Display the first few rows to understand its structure.

### Iteration 2: Explore Data

Visualize the dataset to understand the distribution of `cnt` and its relationship with other features.

- Import visualization libraries: `seaborn` and `matplotlib.pyplot`.
- Visualize the distribution of `cnt` and scatter plots of features against `cnt`.

### Iteration 3: Feature Selection and Preprocessing

#### 3.0 Remove Unnecessary Columns

Remove columns that are not needed for the model, such as `dteday`.

#### 3.1 Normalize Features

Normalize the feature set to ensure that each feature contributes equally to the training process.

### Iteration 4: Build Two Feedforward Neural Networks

In this iteration, you will create and compare **two feedforward neural network models** to predict the total number of bike rentals (`cnt`). One model will be created without any regularization techniques, while the other will include both **dropout** and **L2 regularization**. This will help you understand the impact of these techniques on model performance.

#### 4.1: Build a Feedforward Neural Network without Regularization or Dropout

1. **Define the model architecture**:
   - **Input layer**: Specify the input shape based on the number of features.
   - **Hidden layers**: Include several hidden layers with activation functions (e.g., ReLU).
   - **Output layer**: Use a linear activation function.

2. **Compile the model**:
   - Use an appropriate optimizer (e.g., Adam).
   - Set the loss function to Mean Squared Error (MSE).
   - Include Mean Absolute Error (MAE) as a metric.

#### 4.2: Build a Feedforward Neural Network with Dropout and L2 Regularization

1. **Define the model architecture**:
   - **Input layer**: Specify the input shape based on the number of features.
   - **Hidden layers**: Include hidden layers with both **Dropout** and **L2 regularization**:
     - Apply Dropout layers between the dense layers.
     - Use L2 regularization in the dense layers.

2. **Compile the model**:
   - Use an appropriate optimizer (e.g., Adam).
   - Set the loss function to Mean Squared Error (MSE).
   - Include Mean Absolute Error (MAE) as a metric.

---

### Iteration 5: Train Both Neural Networks

Train **both neural network models** on the training data:

1. Train the model without regularization or dropout using the following parameters:
   - Number of epochs: 100
   - Batch size: 32

2. Train the model with dropout and L2 regularization using the same parameters:
   - Number of epochs: 100
   - Batch size: 32

---

### Iteration 6: Evaluate Both Models

Evaluate the performance of each model on the test data:

1. For the model without regularization or dropout, calculate:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)

2. For the model with dropout and L2 regularization, calculate:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)

Compare the results to see how regularization techniques affect the performance.

---

### Iteration 7: Visualize Training History

Plot and compare the training and validation loss for both models over epochs:

1. Create a plot for the model without regularization or dropout showing:
   - Training Loss
   - Validation Loss

2. Create a plot for the model with dropout and L2 regularization showing:
   - Training Loss
   - Validation Loss

Analyze the plots to diagnose the training process and observe differences.

---

### Iteration 8: Hyperparameter Tuning

Optimize both neural networks by experimenting with different hyperparameters:

1. Tune the number of layers, units per layer, and dropout rates.
2. Use techniques such as **GridSearchCV** from `sklearn`, or perform manual tuning.

---

### Iteration 9: Final Model Evaluation

After hyperparameter tuning, evaluate the final models:

1. Print and compare the final Mean Squared Error (MSE) and Mean Absolute Error (MAE) for both models.
2. Plot and compare the training and validation loss over epochs for both final models.

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

- [ ]()
- [ ](/)
- [ ]()
- [ ]().
- [ ]().

![happy_coding](https://user-images.githubusercontent.com/970858/63899010-c23fc480-c9ea-11e9-84a2-542907e42362.png)
