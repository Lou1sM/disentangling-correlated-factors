from snc_nk import snc_nk
import numpy as np


# replace this with your extracted representations for the whole dataset
# required shape is num_data_points*num_neurons
x = np.random.rand(1000,8)


# replace this with the ground truth for the whole dataset
# required shape is num_data_points*num_generative_factors
y = np.random.randint(5,size=(1000,5))


# classifiers require test data too
x_test = np.random.rand(1000,8)
y_test = np.random.randint(5,size=(1000,5))
print(snc_nk(x, y, x_test, y_test))


# if no test data is passed, 25% will be randomly
# selected as the test set for the classifiers
print(snc_nk(x, y))
