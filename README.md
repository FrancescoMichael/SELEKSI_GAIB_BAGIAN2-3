# SELEKSI_GAIB_BAGIAN2-3
> Seleksi Asisten Lab GaIB ‘22 Bagian 2 (Design of Experiment + Supervised Learning) dan Bagian 3 (Unsupervised Learning)

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Setup](#setup)
* [Usage](#usage)
* [Features](#features)

## General Information
This project is part of the selection process for the Lab GaIB '22 assistant position. It covers both supervised and unsupervised learning techniques, providing a comprehensive exploration of various machine learning models and methods.

## Technologies Used
- Python 3
- Jupyter Notebook

## Setup
Make sure your Python 3 version is appropriate. try running `python3 --version`, if not found, it means your Python 3 installation is not correct.  You can install it from [here](https://www.python.org/downloads/).

Clone this project
```
git clone https://github.com/FrancescoMichael/SELEKSI_GAIB_BAGIAN2-3.git
```

## Usage
If you want to run supervised learning project, from root project folder, run this command: 
```
cd src\supervised-learning
jupyter notebook
```
Open the project at file `main.ipynb`.

If you want to run unsupervised learning project, from root project folder, run this command: 
```
cd src\unsupervised-learning
jupyter notebook
```
Open the project at file `main.ipynb`.

## Features
Supervised Learning (Bagian 2)

✅ kNN 
✅ Logistic Regression
✅ Gaussian Naive Bayes
✅ CART
✅ SVM
✅ ANN

Bonus yang diimplementasikan:
- Logistic Regression: Hinge Loss
- SVM: Polynomial kernel, RBF kernel, Sigmoid kernel
- ANN: tanh activation function, leaky relu activation function, L1 Regularization, L2 Regularization, Adam optimizer, automatic differentiation
- Ensemble Methods: Random Forest

Unsupervised Learning (Bagnian 3)

✅ k-Means
✅ DBSCAN
✅ PCA

Bonus yang diimplementasikan:
- k-Means: Inisialisasi k-means++