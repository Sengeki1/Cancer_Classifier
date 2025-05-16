# Cancer Classifier

A simple breast cancer classifier built using Python 3.10.0 with `scikit-learn`, `pandas`, and `matplotlib`. The classifier uses the Breast Cancer dataset from `sklearn.datasets` and applies both clustering and classification techniques to identify cancer patterns.

## Features

- Loads and processes the Breast Cancer dataset.
- Applies unsupervised learning with `KMeans`.
- Uses supervised learning with `DecisionTreeClassifier` to improve prediction accuracy.
- Visualizes results with `matplotlib`.
- Uses `pandasgui` for interactive data exploration.

## Installation

Make sure you have Python `3.10.0` installed. You can use `pyenv`, `conda`, or `venv` to manage your environment.

```bash
    pip install -r requirements.txt
```

```bash
    python main.py
```

## Results

Using **KMeans clustering**, we achieved an accuracy of `83%`, which demonstrates a solid initial grouping of the data. However, when applying a **supervised learning** approach with a **Decision Tree classifier**, the accuracy improved significantly to `86.5%`. This marked increase highlights the effectiveness of using labeled data for classification in this context. Overall, the results are very promising and indicate that the **Decision Tree** model is well-suited for this task.
