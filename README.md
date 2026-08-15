# Multi-Class ANN Classifier – Palmer Penguins

## Project Overview

This project builds a **Multi-Class Artificial Neural Network (ANN) classifier** using TensorFlow/Keras to predict the species of a penguin.

The model classifies penguins into three species:

* Adelie
* Chinstrap
* Gentoo

## Dataset

The project uses the `penguins_size.csv` dataset.

The target column is:

`species`

The dataset contains both numerical and categorical features, including island, culmen measurements, flipper length, body mass, and sex.

## Data Preprocessing

Before training the ANN, the following preprocessing steps are performed:

1. Invalid values in the `sex` column are replaced with missing values.
2. The target `species` is converted from text labels to numerical labels using `LabelEncoder`.
3. The dataset is divided into training and testing sets using an 80/20 split.
4. Missing numerical values are filled using the median.
5. Numerical features are standardized using `StandardScaler`.
6. Missing categorical values are filled using the most frequent value.
7. Categorical features are converted to numerical features using One-Hot Encoding.

## ANN Architecture

The required neural network architecture is:

```text
Input Layer
    ↓
Dense(32, activation="relu")
    ↓
Dense(16, activation="relu")
    ↓
Dense(n_classes, activation="softmax")
```

For this dataset, `n_classes = 3`.

## Model Compilation

The ANN is compiled using:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Evaluation Metric:** Accuracy

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## Model Training

The model is trained for **30 epochs** with a batch size of 16.

A portion of the training data is used as validation data to monitor model performance during training.

```python
history = model.fit(
    X_train_processed,
    y_train,
    validation_split=0.20,
    epochs=30,
    batch_size=16
)
```

## Model Evaluation

After training, the model is evaluated using the unseen test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    X_test_processed,
    y_test
)

print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
```

The final test accuracy should be reported from the output generated when the notebook is executed.

## Training Visualizations

Two plots are generated to evaluate the training process:

* Training Accuracy vs. Validation Accuracy
* Training Loss vs. Validation Loss

These curves help show how the ANN learns over the 30 training epochs and whether the model is generalizing well to validation data.

## Why Softmax Instead of Sigmoid?

Softmax is used because this is a **multi-class classification problem** with three mutually exclusive penguin species. It produces a probability for each class that sums to 1, whereas a single Sigmoid neuron is mainly suitable for binary classification.

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras

## Files

```text
ann_penguins.ipynb     # Main Google Colab notebook
penguins_size.csv      # Dataset
README.md              # Project documentation
```

## How to Run

1. Open `ann_penguins.ipynb` in Google Colab.
2. Upload `penguins_size.csv`.
3. Run the notebook cells in order.
4. Train the ANN for 30 epochs.
5. Review the final test accuracy.
6. Review the training/validation accuracy and loss plots.

## Conclusion

This project demonstrates how to preprocess mixed numerical and categorical data and build a Multi-Class ANN classifier using Keras. The Softmax output layer allows the trained model to predict one of the three penguin species.
