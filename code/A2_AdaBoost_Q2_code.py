# -*- coding: utf-8 -*-
"""
Created On : Fri Oct 15 2021
Last Modified : Fri Oct 22 2021
Course : MSBD5002 
Assignment : Assignment 02 Question 02

Remark:
    - Base Classifier of choice for the Adaboost - Stump (decision tree of max depth of 1)
    - Base Classifiers are either x < v or x > v
    - if x < v,  y_predict = 1 otherwise y_predict = -1 (user define function, stump_tree())
    - or if x > v,  y_predict = 1 otherwise y_predict = -1 (user defined function, stump_tree_opposite())
    - Importance (weight) of each Base Classifier represented by - 'alpha' variable
"""

import pandas as pd
import numpy as np
import math


# The X input, Y input and the initial weights of each data
# Values in this attribute are Numerical Data
x_list_og = [0,1,2,3,4,5,6,7,8,9] 
y_list_og = [1,1,-1,-1,-1,1,1,-1,-1,1]
x_weights_og = [0.1]*10 

################# User-defined function #################
# Define a function for the stump that will be the base classifier for the adaboost algorithm 
# This function considers if x < v (v being the threshold)
# if x < v,  y_predict = 1 otherwise y_predict = 0
def stump_tree(x_values,y_values, threshold):
    """ Create a stump (decision tree of max depth of 1)
        if x < v,  y_predict = 1 otherwise y_predict = -1 """ 
    leftN_1 = [] # This will return the list of x correctly classified as 1
    leftN_incorrect = [] # This will return the list of x incorrectly classified as 1
    rightN_minus1 = [] # This will return the list of x correctly classified as -1
    rightN_incorrect = [] # This will return the list of x incorrectly classified as -1
    for i in range(0,len(x_values)):
        x = x_values[i]
        y = y_values[i]
        if x_values[i] < threshold: # Left Node (For 1)
            # Collecting all the x in left node when x < v
            if y == 1:
                leftN_1.append(x)
            else:
                leftN_incorrect.append(x)
        else: # Right Node (For -1)
            # Collecting all the x in right node when otherwise
            if y == -1:
                rightN_minus1.append(x)
            else:
                rightN_incorrect.append(x)  
    return leftN_1, leftN_incorrect, rightN_minus1, rightN_incorrect


# Define a function for the stump that will be the base classifier for the adaboost algorithm 
# This function considers if x > v (v being the threshold)
# if x > v,  y_predict = 1 otherwise y_predict = 0
def stump_tree_opposite(x_values,y_values, threshold):
    """ Create a stump (decision tree of max depth of 1) 
        if x > v,  y_predict = 1 otherwise y_predict = -1 """ 
    leftN_1 = [] # This will return the list of x correctly classified as 1
    leftN_incorrect = [] # This will return the list of x incorrectly classified as 1
    rightN_minus1 = [] # This will return the list of x correctly classified as -1
    rightN_incorrect = [] # This will return the list of x incorrectly classified as -1
    for i in range(0,len(x_values)):
        x = x_values[i]
        y = y_values[i]
        if x_values[i] > threshold: # Left Node (For 1)
            # Collecting all the x in left node when x > v
            if y == 1:
                leftN_1.append(x)
            else:
                leftN_incorrect.append(x)
        else: # Right Node (For -1)
            # Collecting all the x in right node when otherwise
            if y == -1:
                rightN_minus1.append(x)
            else:
                rightN_incorrect.append(x)              
    return leftN_1, leftN_incorrect, rightN_minus1, rightN_incorrect


# Define a function for calculating the error rate for each base classifier during fitting
def determine_error_rate(leftN_incorrect, rightN_incorrect, row_x, weights_x):
    """ Function is used to determine the error rate for each base classifier
        during training of the base classifier using the sampled dataset.
        Formula used = (sum of misclassified weights)/ (length of dataset, N) """
    incorrect_list = leftN_incorrect + rightN_incorrect
    incorrect_list = np.unique(incorrect_list) # Stores the x that were misclassified
    weights_determined = []
    for incorrect_x_index in range(0, len(row_x)):
        x_check = row_x[incorrect_x_index]
        if x_check in incorrect_list:
            # If a particular vx was misclassified, store the weight of that x value 
            weights_determined.append(weights_x[incorrect_x_index])  
    return sum(weights_determined)/len(weights_x) # sum up the weights that were misclassified and divide over length of the dataset N
#########################################################




################# Adaboost Algorithm #################
### The Adaboost Algorithm will iterate over multiple best base (weak) classifiers
### until all the Y is predicted accurately (Using a While loop to achieve this) 
## Part 1 : 
# - Initialize the Dataframe containing the original X and Y and the updated Weights (df_x_y_table)
# - Generate a Sample of Dataset (df_x_y_sample) from the Original Dataset ("df_x_y_table")
# - Generate a set of thresholds depending on the Sampled Dataset 
## Part 2:
# - Iterate over all the set of thresholds and determine the Best Base Classifier 
# - Multiple Base Classifiers will be trained with the Sampled Dataset
# - The Base Classifier could be either (if x<=v, y=1) or (if x>=v, y=1) 
# - The Best Base Classifier is selected if it has the lowest error rate (from the Sampled Dataset)
## Part 3:
# - The Selected Base Classifier (from Part 2) will be used to predict the Original Dataset (X Label)
# - The Number of Misclassified X (Incorrectly Predicted Y) is recorded
# - The Error Rate and Alpha (Important of Classifier) of the Selected Base Classifier will be calculated 
# - After that, the weights of the Original Dataset will be updated based on the Misclassified X
## Part 4:
# - Sum up the Prediction of every Selected Base Classifier at every iteration
# - Sign Function is used to output the Latest Y Prediction by the Adaboost algorithm 
# - This Part will check if the Final Y Prediction has an Accuracy of 1  (accuracy_flag)
# - The Adaboost algorithm will end the 'while loop' when total Y Predictions are accurate (accuracy_flag)
######################################################

# The Variables below is to initialize the Original Dataset
x_list = x_list_og
y_list = y_list_og
x_weights = x_weights_og
# The Variable below is to be stored with the predicted y values in a dataframe 
dict_x_y_predict = {'X':x_list_og,'Y':y_list_og}
alpha_list = [] # This variable contains the importance (weight) of each base classifier 
# Other Variables
latest_sum_predicted_y = [0]*10 # this is to initialize the current summation of the Predicted Y 
accuracy_flag = 0 # This flag is to stop the While loop when the final predicted output is 100% accurate
weak_clf_i = 0  # This variable is to record the number Base Classifiers Selected

### The Following Code is the Adaboost Algorithm (Refer to Description Above as a Guide)
while accuracy_flag != 1: # Continue Iterating until all Y values are predicted accurately
    print("") 
    print("="*60) 
    print("Base Classifier/ Weak Classifier (Stump) No. "+str(weak_clf_i+1))
    print("="*60) 
    
    ### Part 1: ###
    ### Initialize the weights of the original dataset
    # The df_x_y_table dataframe variable will be used to store the updated weights at every iteration 
    print('\nTable of Original Dataset with Current Weight (Updated)')
    print("-"*60) 
    dict_x_y_table = {'X':x_list_og, 'Y':y_list_og, 'weights':x_weights}
    df_x_y_table = pd.DataFrame(dict_x_y_table) # This variable will be referenced throughout this algorithm 
    print(df_x_y_table)  # Print the DataFrame Table displaying the Original DataSet

    ### Sample the dataset to be used to train the base classifier
    print("\nTable of Sampled Dataset for Training the {}th Base Classifiers".format(str(weak_clf_i+1)))
    print("-"*60) 
    ## Determine the sampled data x using the function np.random.choice()
    weights_current = df_x_y_table['weights'].tolist()
    x_to_be_sampled = df_x_y_table['X'].tolist()
    y_to_be_sampled = df_x_y_table['Y'].tolist()
    sampled_data_x = np.random.choice(a=x_to_be_sampled,size= (10), p=weights_current)
    
    ## After sampling the data x (above): 
    # - Determine the weights of the corresponding sampled data x
    # - Determine the y of the corresponding sampled data x
    # - and Store the sampled x, y and weights information in the dataframe (df_x_y_sample)
    sampled_weights = [] 
    sampled_data_y=[]
    for sample in sampled_data_x:
        # identify the corresponding y and weights associated with the sampled x value
        index_sample = x_to_be_sampled.index(sample)
        sampled_weights.append(weights_current[index_sample])
        sampled_data_y.append(y_to_be_sampled[index_sample])
    # store the sampled x, y and weights into the dataframe  
    dict_x_y_sample = {'X':sampled_data_x, 'Y':sampled_data_y, 'weights':sampled_weights}
    df_x_y_sample = pd.DataFrame(dict_x_y_sample)
    print(df_x_y_sample) # Print the DataFrame Table displaying the Sampled DataSet
    
    ### Determine the set of thresholds, v to be used by each base classifier to fit the sampled dataset  
    # - The threshold for each iteration depends on the sampled dataset 
    x_adjust_threshold = df_x_y_sample['X'].tolist()
    threshold_list = np.arange(min(x_adjust_threshold)+0.5 , max(x_adjust_threshold)+1.5 , 1).tolist()
    #! Uncomment to display the Threshold for each Iteration !#
    # print("\nThreshold for the Current Base Classifiers") 
    # print(threshold_list) 
    
    
    
    ### Part 2: ###
    ### The Following section will iterate over every threshold to find the Best Base Classifier
    # The following variables is to store the x values depending on whether it was correctly classified
    leftnode_c_list = [] # store a list of all the correctly classified x
    leftnode_incorrect_list = []  # store a list of all the incorrectly classified x
    rightnode_c_list = []  # store a list of all the correctly classified x
    rightnode_incorrect_list = []  # store a list of all the incorrectly classified x
    # The following variables is to store the key information on each classifier (threshold, error and type of classsifier)
    threshold_v_list = [] # store the threshold that was used by each classifier
    stump_error_list = [] # store the error rates of each classifier after fitting 
    stump_type= [] # store the type of base classifier for each classifier (either x < v or x > v ) in a list
    for v in threshold_list:
        #############################################################################
        ## Part (A) Determine the error rate for the base classifier of type x < v
        ## Part (A) Uses Function stump_tree
        threshold_v_list.append(v) 
        stump_type.append(['<'])
        # input the sampled data into the base classifier (x < v)
        leftnode_c, leftnode_incorrect, rightnode_c, rightnode_incorrect = stump_tree(sampled_data_x,sampled_data_y,v)   
        
        # store the correctly classsified and misclassified x 
        leftnode_c_list.append(leftnode_c)
        leftnode_incorrect_list.append(leftnode_incorrect)
        rightnode_c_list.append(rightnode_c)
        rightnode_incorrect_list.append(rightnode_incorrect)

        # Determine the error rate for each base classifier (x < v)
        input_x_weights = dict_x_y_sample['weights']
        input_x_row = dict_x_y_sample['X']
        error_rate_base_c = determine_error_rate(leftnode_incorrect, rightnode_incorrect, input_x_row, input_x_weights)
        # store the error rate for the base classifier (x < v)
        stump_error_list.append(error_rate_base_c)

        #############################################################################
        ## Part (B) Determine the error rate for the base classifier of type x > v
        ## Part (B) Uses Function stump_tree_opposite
        threshold_v_list.append(v)
        stump_type.append(['>'])
        # input the sampled data into the base classifier (x > v)
        leftnode_c, leftnode_incorrect, rightnode_c, rightnode_incorrect = stump_tree_opposite(sampled_data_x,sampled_data_y,v)   
        
        # store the correctly classsified and misclassified x 
        leftnode_c_list.append(leftnode_c)
        leftnode_incorrect_list.append(leftnode_incorrect)
        rightnode_c_list.append(rightnode_c)
        rightnode_incorrect_list.append(rightnode_incorrect)
        
        # Determine the error rate for each base classifier (x > v)
        input_x_weights = dict_x_y_sample['weights']
        input_x_row = dict_x_y_sample['X']
        error_rate_base_c = determine_error_rate(leftnode_incorrect, rightnode_incorrect, input_x_row, input_x_weights)
        # store the error rate for the base classifier (x > v)
        stump_error_list.append(error_rate_base_c)
    
    
    ### Determine the base classifier (stump) with the lowest error rate 
    # find the lowest error rate
    stump_error_min = min(stump_error_list) 
    stump_error_min_index = stump_error_list.index(stump_error_min) # find the index 
    # find the threshold that correspond to the lowest error rate
    threshold_v_min = threshold_v_list[stump_error_min_index] 
    # find the type of base classifier that correspond to the lowest error rate
    threshold_type_min = stump_type[stump_error_min_index] 
    print("\nInformation on the Selected Base Classifier after Training with Sampled Dataset")
    print("-"*60) 
    print("Lowest Error Rate: {} \nThreshold Selected: {}".format(stump_error_min,threshold_v_min ))
    print("Type of Base Classifier Selected((if x<v, y=1) or (if x>v, y=1)): {}".format(threshold_type_min))
    ##! Uncomment to Display the DataFrame Table of all error rate for all threshold !##
    # dict_v_error = {'Threshold': threshold_v_list, 'Error Rate': stump_error_list,'Left':leftnode_incorrect_list,'Right':rightnode_incorrect_list, 'Type':stump_type}
    # df_v_error = pd.DataFrame(dict_v_error) # Purpose is to display a table of all error rate of the threshold
    # print("\nThreshold and Error Rate")
    # print(df_v_error) 



    ### Part 3: ###
    ### This Section will use the Selected Base Classifier to Predict the Y values of Original Dataset
    # Following variables are used for storing x values that were either correctly or incorrectly classified
    leftog_correct = []
    leftog_incorrect = []
    rightog_correct = []
    rightog_incorrect = []
    classification = [] # This list will store labels 'yes' or 'no' (Purpose to Print in "df_x_y_classification" DataFrame Table)
    y_list_predict = [] # Store the Predicted Y
    # Following variables will be referenced throughout this section
    x_list_p = df_x_y_table['X'].tolist() # x value from original dataset
    y_list_p = df_x_y_table['Y'].tolist() # y value from original dataset
    # Determine the correct and incorrect Y Classification from the Original Dataset
    for i in range(0,len(x_list_p)):
        x = x_list_p[i]
        #############################################################################
        # Part (A) Determine the predicted y value with the base classifier of type (x < v, predicted y = 1)
        if threshold_type_min[0] == '<': 
            # If the Selected Base Classifer is type (x < v, predicted y = 1), run this code
            if x < threshold_v_min: # Left Node (For 1)
            # Collecting all the x in left node when x < v
                if y_list_p[i] == 1:
                # If y is predicted correctly (as 1)
                    leftog_correct.append(x)
                    classification.append('Yes')
                    y_list_predict.append(1)
                else: 
                # If y was misclassified (as -1)
                    leftog_incorrect.append(x)
                    classification.append('No')
                    y_list_predict.append(1)
            else: # Right Node (For -1)
            # Collecting all the x in right node when x < v
                if y_list_p[i] == -1: 
                # If y is predicted correctly (as -1)
                    rightog_correct.append(x)
                    classification.append('Yes')
                    y_list_predict.append(-1)
                else: 
                # If y was misclassified (as 1)
                    rightog_incorrect.append(x)    
                    classification.append('No')
                    y_list_predict.append(-1)
        #############################################################################
        # Part (B) Determine the predicted y value with the base classifier of type (x > v, predicted y = 1)
        if threshold_type_min[0] == '>':
            # If the Selected Base Classifer is type (x > v, predicted y = 1), run this code
            if x > threshold_v_min: # Left Node (For 1)
            # Collecting all the x in left node when x > v
                if y_list_p[i] == 1: 
                # If y is predicted correctly (as 1)
                    leftog_correct.append(x)
                    classification.append('Yes')
                    y_list_predict.append(1)
                else: 
                # If y was misclassified (as -1)
                    leftog_incorrect.append(x)
                    classification.append('No')
                    y_list_predict.append(1)
            else: # Right Node (For -1)
            # Collecting all the x in right node when x > v
                if y_list_p[i] == -1: 
                # If y is predicted correctly (as -1)
                    rightog_correct.append(x)
                    classification.append('Yes')
                    y_list_predict.append(-1)
                else: 
                # If y was misclassified (as 1)
                    rightog_incorrect.append(x)    
                    classification.append('No')
                    y_list_predict.append(-1)                

    ##! Uncomment to Display the DataFrame Table !##
    # ### Print and Display the Original Dataset with its Predicted Y and Accuracy of Y Prediction       
    # print("\nDisplay Original Dataset with the Predicted Y and a Column to indicate if the Y Prediction was correct")
    # dict_x_y_classification = {'X':x_list_p, 'Y':y_list_p, 'Y(Predict)':y_list_predict,'Classification':classification}
    # df_x_y_classification = pd.DataFrame(dict_x_y_classification)
    # print(df_x_y_classification)
        
    ### Append the incorrectly classified rows into a separete list to be referred to later on in the code
    ## -  this variable will be used to update the weights of the original dataset
    incorrect_classified_x = leftog_incorrect + rightog_incorrect
    incorrect_classified_x = np.unique(incorrect_classified_x)   
    print("\nInformation on the Selected Base Classifier after Predicting with the Original Dataset")
    print("-"*60) 
    print("Incorrectly Classified X value: {}".format(incorrect_classified_x))
   
    ### Determine the Error Rate of the Selected Base Classifier after predicting the Y with the Original Dataset   
    # - Error Rate based on the misclassified data from original dataset
    # - This Error Rate will be used to calculate the alpha (Importance of the Classifier/ Weight of Classifier)
    # - Formula used = (sum of misclassified x weights)/ (length of dataset, N)
    all_weights_mis = df_x_y_table['weights'].tolist()
    weights_to_be_sum = []
    x_list_check = df_x_y_table['X'].tolist()
    for x_mis_index in range(0, len(x_list_check)):
        x_mis = x_list_check[x_mis_index]
        if x_mis in incorrect_classified_x:
            weights_to_be_sum.append(all_weights_mis[x_mis_index])
    error_rate = sum(weights_to_be_sum)/10
    print("Error Rate for the Selected Base Classifier: {}".format(error_rate))
  
    ### Determine the Alpha (Importance of the Classifier/ Weight of Classifier)
    if error_rate != 0 and error_rate != 1:
        alpha = (0.5)*math.log(((1-error_rate)/(error_rate)))
    else:
        if error_rate == 0:
            error_term = 0.001
            alpha = (0.5)*math.log(((1-error_term)/(error_term)))
        if error_rate == 1:
            error_term = 0.999
            alpha = (0.5)*math.log(((1-error_term)/(error_term)))
    print("Importance of the Selected Base Classifier (alpha): {:.3f}".format(alpha) )

    ### Record the results of each Predicted Y by the Selected Base Classifier
    # - The purpose of this section is to print the results as a dataframe table at the end of the Adaboost algorithm 
    classifier = 'C'+str(weak_clf_i+1)
    dictadd = {classifier:y_list_predict}
    dict_x_y_predict.update(dictadd) # To be printed as a DataFrame Table at the end
    
    ### Record the alpha (Importance of the Classifier/ Weight of Classifier) of each Selected Base Classifiers 
    alpha_list.append(alpha) # For printing the final expression at the end of the Adaboost algorithm 
       
    ### This Section will adjust the weights of the Original Dataset 
    ## - Weights are adjusted using the formula (previous weights)*exponential(alpha or - alpha) 
    ## - Adjust the weights of the original dataset  (Not yet normalized)
    weights_list_n  = df_x_y_table['weights'].tolist() # Extract the latest weights of the dataset
    x_list_n = df_x_y_table['X'].tolist() # Extract the x value from original dataset
    all_weights_new = []
    for x_update_index in range(0, len(x_list_n)):
        x_update = x_list_n[x_update_index]
        if x_update in incorrect_classified_x:
            # If the x was Incorrectly Classified, Weights will be increased
            weights_i_new = (weights_list_n[x_update_index])*math.exp(alpha)
            all_weights_new.append(weights_i_new)
        else:
            # If the x was Correctly Classified, Weights will be decreased
            weights_i_new = (weights_list_n[x_update_index])*math.exp(-alpha)
            all_weights_new.append(weights_i_new)
    
    ## - After calculating the updated weights for each x value
    ## - This section will Normalize the Weights across all x 
    sum_weights_notnormal = sum(all_weights_new)
    all_weights_normalize = [ x/sum_weights_notnormal for x in all_weights_new]
     
    ##! Uncomment to display the old weights and the new weights together !##
    # ### Print and Display the old weights and the new weights 
    # dict_for_disp = {'X':x_list_og, 'Y':y_list_og, 'weights':weights_list_n, 'weights(new)':all_weights_new, 'weights(normal)':all_weights_normalize}
    # df_for_disp = pd.DataFrame(dict_for_disp)
    # print("\nDisplay the old weights and new weights")
    # print(df_for_disp)
    
    ### Update the New and Normalized weights to the Original Dataset
    x_weights = all_weights_normalize
    
    ### Records the number of Base Classifier Selected after every Iteration is Done
    weak_clf_i += 1 


    
    ### Part 4: ###
    ### This Section will keep track and sum up the Y Prediction after every Base Classifier is Selected 
    ## - First, the alpha will be multiplied by the Predicted Y
    ## - Then added to the previous Summation of (alpha*PredictedY) 
    ## - the "alpha" in this section refers to the  Weight of current Selected Base Classifier
    ## - the "y_list_predict" in this section refers to the Predicted Y from the current Selected Base Classifier
    current_prediction= y_list_predict
    current_prediction_alpha = [ alpha*y for y in current_prediction]
    sum_predicted_y = [x + y for x, y in zip(current_prediction_alpha, latest_sum_predicted_y)] # Sum up the current with previous summation
    latest_sum_predicted_y = sum_predicted_y # Update and store the latest summation for the next Iteration (Selected Base Classifier)

    ### Input the current Summation of the Selected Base Classifier so far into the Sign Function   
    # - use Sign Function to determine the Final Prediction of every Summation of Selected Base Classifier
    final_predicted_y = np.sign(sum_predicted_y)

    ### Check the Accuracy Score of the Final Prediction of Summation of the Selected Base Classifier
    # - only concern with Accuracy Score being = 1 (which is when all the Y were predicted correctly)
    # - did not use the metrics.accuracy_score to avoid using sklearn library (Note: Adaboost Library Not Allowed)
    final_predicted_y_list = list(final_predicted_y)
    if final_predicted_y_list == y_list_og:
        accuracy_flag = 1
    else:
        accuracy_flag = 0




##############################################################
### Print and Display Final Prediction and Key Information ###
## Print the Number of Base Classifiers 
print("\n\n") 
print("="*60) 
print("Print and Display Final Prediction and Final Expression of Adaboost")
print("="*60) 
print("\nNumber of Base Classifiers Required to Accurately Predict all Y Values:")
print("Number of Base Classifiers : {}".format(weak_clf_i))

## Print and Display Table of Predicted Y for each Classifier and the Final Prediction
final_dictadd = {'Final':final_predicted_y_list }
dict_x_y_predict.update(final_dictadd)
df_x_y_predict = pd.DataFrame(dict_x_y_predict)
print("\nTable to Display all the Base Classifiers (C#) Y Prediction and Final Prediction")
print("-"*60)
print(df_x_y_predict)

## Print and Display Final Expression of the Strong CLassifier 
print('\nDisplay the Final Expression of the Strong Classifier (Alpha (Importance) & Base Classifier): ')
classifier_number = 1
alpha_list_rounded = [round(element, 3) for element in alpha_list]
for aa in range(0, len(alpha_list_rounded)): 
    if aa == 0:
        final_expression_adaboost = "C*(x) = sign[ " + str(alpha_list_rounded[aa])+"*"+"Classifier"+str(classifier_number)
    else:
        final_expression_adaboost += " + " + str(alpha_list_rounded[aa])+"*"+"Classifier"+str(classifier_number)
    classifier_number += 1
print(final_expression_adaboost + ' ]')
print("-"*60) 
##############################################################












