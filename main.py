"""
Name:                   
    Nima DindarSafa
    Samira Abedini

Student ID:
    7072844
    7072848

Email:
    nima.dindarsafa@gmail.com
    samiraabedini150@gmail.com
"""

# Assignment 1

# Excercise 1

# Question 1
from assignment1_1_1 import create_dataset

input_shape0 = (100,2)
input_shape1 = (100,2)

clus1_shift_vec = [-2,-2]
clus2_shift_vec = [2,2]

points, labels = create_dataset(input_shape0, input_shape1, clus1_shift_vec, clus2_shift_vec)

print(f"The genereated dataset randomly shuffled points are:\n {points} \n")
print(f"The shape of generated data points is: {points.shape}\n")

print(f"The respective labels for dataset with the same indeces for points are:\n {labels}\n")
print(f"The shape of generated labels is: {labels.shape}")

# Question 2
from assignment1_1_2 import gen_xor, gen_label

input_shape = (4,2)

xor_array = gen_xor(input_shape)
print(f"The generated XOR data is: \n {xor_array}")
print(f"The shape of XOR dataset: \n {xor_array.shape} \n")


labels_xor = gen_label(xor_array)
print(f"The respective labels for the XOR dataset is: \n {labels_xor}")
print(f"The shape of labels: \n {labels_xor.shape}")

# # Question 3
from assignment1_1_3 import plot
from assignment1_1_1 import gen_array, shift_mean

cluster1, cluster2 = gen_array(input_shape0), gen_array(input_shape1)

cluster1 = shift_mean(cluster1, clus1_shift_vec)
cluster2 = shift_mean(cluster2, clus2_shift_vec)

plot(cluster1, cluster2)

# Excercise 2

# Question 1
from assignment1_2_1 import simple_svc_linear_decision_boundry

model = simple_svc_linear_decision_boundry()

classifier_simple = model.fit(points, labels.ravel())
print(f"The score of fitting SVM to our data is {classifier_simple.score(points, labels.ravel())}.\nTo see the implementation please refer to source code.")

# Question 2
from assignment1_2_2 import plot_decision_boundry

grid_step = 0.01

title1 = "Decision boundry for simple SVM model on randomly generated data"
plot_decision_boundry(points, labels.ravel(), grid_step, classifier_simple, title= title1)

# Question 3
from assignment1_2_3 import grid_search_cv, randomized_search

import warnings
from sklearn.exceptions import FitFailedWarning


parameters_grid = {
    'penalty':['l2','l1'],
    'loss':['hinge', 'squared_hinge'],
    'C':[0.1,0.5,1,5,10],
}

# Some combinations of parameters_grid is not supported by LinearSVC hence it throws warning messages in terminal
# In order to ignore the warnings we have put them in warning catcher. 
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    grid_search = grid_search_cv(model, parameters_grid, cv=2)

    grid_clf_rand = grid_search.fit(points, labels.ravel())
    grid_clf_xor = grid_search.fit(xor_array, labels_xor.ravel())

print(f"Best parameters achieved by grid search is:\n{grid_search.best_params_}\n")
print(f"Best accuracy achieved by grid search is:{grid_search.best_score_}\n")

title2 = "Decision boundry for best model of grid search on randomly generated data"
title3 = "Decision boundry for best model of grid search on XOR data"

plot_decision_boundry(points, labels.ravel(), grid_step, grid_clf_rand, title=title2)
plot_decision_boundry(xor_array, labels_xor.ravel(), grid_step, grid_clf_xor, title=title3)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FitFailedWarning)
    random_search = randomized_search(model,parameters_grid, num_iter= 10)

    random_clf_rand = random_search.fit(points, labels.ravel())
    random_clf_xor = grid_search.fit(xor_array, labels_xor.ravel())

print(f"Best parameters achieved by random search is:\n{random_search.best_params_}\n")
print(f"Best accuracy achieved by random search is:{random_search.best_score_}\n")

title4 = "Decision boundry for best model of random search on randomly generated data"
title5 = "Decision boundry for best model of random search on XOR data"

plot_decision_boundry(points, labels.ravel(), grid_step, random_clf_rand, title=title4)
plot_decision_boundry(xor_array, labels_xor.ravel(), grid_step, random_clf_xor, title=title5)

# Excercise 3

# Question 1
"""
The points in the first dataset are linearly seperable as they are created with this purpose. However, the XOR dataset does not
have such feature. As a result, the model did not work for XOR dataset. ALso, the points of XOR are randomly generated 
ints between 0 and 1, hence in some runs of the script there are labels only of one class and the script throws error. 
"""

# Question 2
"""
According to results of our code, decision boundry is NOT unique. First of all data is randomly generated and without a seed,
at each run it will create a new set of data. It is obvious that in this case, at each run new decision boundry will be created.
Even if that is not the case, decision boundry that fits to data is highly dependent to hyperparameters of model. As we see the 
decision boundry of simple linearsvc, grid search and random search is not the same. Since each of the mentioned models have their own set
of hyperparameter, the new model will yield a different decision boundry. As a result, it is safe to claim that decision boundry
is NOT unique.
"""
# Question 3
from assignment1_1_1 import create_dataset
from assignment1_2_1 import simple_svc_linear_decision_boundry
from assignment1_2_2 import plot_double_decision_boundry

import numpy as np

input_shape0_outl = (8,2)
input_shape1_outl = (8,2)

clus1_shift_vec = [-2,-2]
clus2_shift_vec = [2,2]

outlier_points, outlier_labels = create_dataset(input_shape0_outl, input_shape1_outl, clus1_shift_vec, clus2_shift_vec)


outlier_labels *= -1

noisy_points = np.vstack((points, outlier_points))
noisy_labels = np.vstack((labels, outlier_labels))

model = simple_svc_linear_decision_boundry()

classifier_noisy = model.fit(noisy_points, noisy_labels.ravel())
print(f"The score of fitting SVM to outlier data is {classifier_noisy.score(noisy_points, noisy_labels.ravel())}.\n")

title6 = "Decision boundry simple vs. noisy"
# The line for noisy data is plotted by dotted line. 
plot_double_decision_boundry(noisy_points, noisy_labels.ravel(),
                             0.01,
                             classifier_simple,
                             classifier_noisy, 
                             title=title6)
"""
The inclination of the predicted decision boundry is changed and the line is shifted after adding the random outliers since noises 
can impact a linear model like LinearSVC significantly. The model tries to also fit the mislabeled data and this is why the decision
boundry tries to become more close to the cluster with more mislabeled data points. This can be tested by changing the first element of
input_shape0_outl and input_shape1_outl to try different number of outlier points.
"""