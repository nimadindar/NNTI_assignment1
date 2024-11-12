# NNTI Assignment 1


This repository contains implementations and solutions to the exercises outlined in the NNTI Assignment 1. The primary concepts covered include data generation and visualization using NumPy and Matplotlib, classification using Scikit-learn's SVM, and decision boundary analysis. Below, you'll find an overview of each implemented task.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Implemented Concepts](#implemented-concepts)
  - [Exercise 1: Data Generation and Visualization](#exercise-1-data-generation-and-visualization)
  - [Exercise 2: Classification and Decision Boundary](#exercise-2-classification-and-decision-boundary)
  - [Exercise 3: Analysis of Model Performance](#exercise-3-analysis-of-model-performance)


---

## Prerequisites
- Python 3.x
- Libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/nimadindar/NNTI_assignment1.git

Install required packages:
bash
Copy code
pip install numpy matplotlib scikit-learn
Navigate through the exercises using the provided scripts.
Implemented Concepts
Exercise 1: Data Generation and Visualization
Objective: Generate and visualize data clusters using vectorized NumPy operations.

Task 1: Generate two sets of 100 linearly separable points with classes -1 and 1.
Points are centered at [-2, -2] and [2, 2].
Task 2: Generate an XOR dataset with four specific points.
The class of each point is determined by the XOR operation of its coordinates, with labels changed from [0, 1] to [-1, 1].
Task 3: Visualize the generated clusters using matplotlib.
Exercise 2: Classification and Decision Boundary
Objective: Use Scikit-learn's LinearSVC to create and optimize a linear decision boundary.

Task 1: Fit a linear SVM model to classify the generated data.
Task 2: Visualize the decision boundary created by the SVM model.
Task 3: Optimize and visualize the decision boundary for both datasets (linearly separable and XOR).
Exercise 3: Analysis of Model Performance
Objective: Analyze the decision boundary and explore its behavior.

Task 1: Observe and explain the model's performance on both datasets.
Task 2: Determine the uniqueness of the decision boundary.
Task 3: Introduce outliers by flipping the class of 8 randomly selected points and observe changes in the decision boundary.
