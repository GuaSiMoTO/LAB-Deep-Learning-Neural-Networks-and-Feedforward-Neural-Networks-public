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
    cd LAB-Supervised-Learning-Regression-Models
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

### Iteration 4: Build a Feedforward Neural Network

Create a feedforward neural network model to predict the total number of bike rentals (`cnt`).

- Define the model architecture:
  - Input layer
  - Hidden layers with activation functions (e.g., ReLU)
  - Output layer with a linear activation function

- Compile the model with an appropriate optimizer and loss function.

### Iteration 5: Train the Neural Network

Train the feedforward neural network on the training data.

- Use suitable parameters such as the number of epochs and batch size.

### Iteration 6: Evaluate the Model

Evaluate the performance of the neural network model on the test data.

- Calculate the Mean Squared Error (MSE) and Mean Absolute Error (MAE).

### Iteration 7: Visualize Training History

Plot the training and validation loss over epochs to diagnose the training process.

### Iteration 8: Hyperparameter Tuning

Optimize the neural network by tuning hyperparameters such as the number of layers, number of units per layer, and activation functions.

- Use techniques like GridSearchCV or RandomizedSearchCV from `sklearn`, or manual tuning.

### Iteration 9: Final Model Evaluation

After tuning and comparison, evaluate the final feedforward neural network model.

- Print and compare MSE and other relevant metrics.
- Plot the training and validation loss over epochs to diagnose the training process.

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
