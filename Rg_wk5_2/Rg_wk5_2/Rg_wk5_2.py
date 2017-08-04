import graphlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    #[optional] Seaborn makes plot nicer
    import seaborn;
except ImportError:
    pass;

sales = graphlab.SFrame('D:\\ML_Learning\\UW_Regression\\Week5\\kc_house_data.gl\\')
sales['floors'] = sales['floors'].astype(int)


#return 1+h0(xi)+h1(xi)+...., and output
def get_numpy_data(data_sframe,features,output):
    data_sframe['constant'] = 1
    features = ['constant'] + features
    features_sframe = data_sframe[features]
    features_matrix = features_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()
    return (features_matrix,output_array)


#calculate the dot product of w*h
def predict_output(feature_matrix,weights):
    pred = np.dot(feature_matrix,weights)
    return pred

#normalize features by 2-norm
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    feature_matrix = feature_matrix / norms
    return (feature_matrix, norms)

features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
print norms

ro = []
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)
weights = np.array([1., 4., 1.])
prediction = predict_output(simple_feature_matrix,weights)
errors =  output - prediction
for i in xrange(len(weights)):
    feature_i = simple_feature_matrix[:,i]
    #ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
    errors = errors + weights[i] * feature_i
    ro.append(np.dot(feature_i,errors))

    
#lasso coordinate descent
def lasso_coordinate_descent_step(i,feature_matrix,output,weights,l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix,weights)
    feature_i = feature_matrix[:,i]
    errors = output - prediction
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    errors = errors + weights[i] * feature_i
    ro_i = np.dot(feature_i,errors)
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.    
    return new_weight_i

import math
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]]), 
                                   np.array([1., 1.]), np.array([1., 4.]), 0.1)
#above is correct


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights
    converged = False
    while not converged:
        weights_change = np.zeros(len(weights))
        for i in range(len(weights)):
            old_weights_i = weights[i] # remember old value of weight[i], as it will be overwritten
            # the following line uses new values for weight[0], weight[1], ..., weight[i-1]
            #     and old values for weight[i], ..., weight[d-1]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            # use old_weights_i to compute change in coordinate
            #weights_change.append(abs(old_weights_i-weights[i]))
            weights_change[i]= abs(weights[i]-old_weights_i)
        #weights_change = np.asfarray(weights_change)        
        #max_change = max(weights_change)
        if np.max(weights_change) < tolerance:
            converged = True
    return weights

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features
weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print weights

#Calculate rss
def get_rss(pred, output):
    rss = pred - output
    rss = rss * rss
    return rss.sum()
pred = predict_output(normalized_simple_feature_matrix,weights)
rss = get_rss(pred,output)
print rss

train_data,test_data = sales.random_split(.8,seed=0)
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']

l1_penalty = 1e7
tolerance = 1.0
initial_weights = np.zeros(len(all_features)+1)
(all_feature_matrix, output) = get_numpy_data(train_data, all_features, my_output)
(normalized_all_feature_matrix, all_norms) = normalize_features(all_feature_matrix) # normalize features
weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
print weights1e7
l1_penalty = 1e8
weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
#print weights1e8
l1_penalty = 1e4
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
#print weights1e4
weights1e7_norm = weights1e7 / all_norms
weights1e8_norm = weights1e8 / all_norms
weights1e4_norm = weights1e4 / all_norms
print weights1e7_norm[3]

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')
pred1e7 = predict_output(test_feature_matrix,weights1e7_norm)
rss_1e7 = get_rss(pred1e7,test_output)
pred1e4 = predict_output(test_feature_matrix,weights1e4_norm)
rss_1e4 = get_rss(pred1e4,test_output)
pred1e8 = predict_output(test_feature_matrix,weights1e8_norm)
rss_1e8 = get_rss(pred1e8,test_output)
print rss_1e7
print rss_1e8
print rss_1e4

