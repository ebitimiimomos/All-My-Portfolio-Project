import numpy as np
import pandas as pd

#Question 2
def perceptronTrain(train, maxIter):
    '''
    function trains a binary classifier using the perceptron algorithm.
    '''

    # Initialize weighteight vector and bias term
    n_X = len(train[0][0]) # Get the number of features in the input
    weight = [0.0] * n_X # Initialize the weight vector as a list of zeros
    bias = 0.0 # Initialize the bias term to zero

    # Loop over training set for a maximum numbiaser of iterations
    for iter in range(maxIter):
        # Loop over all training examples
        for X, y in train:
            # Compute activation code as the dot product of the weight vector and the input,
            # plus the bias term
            a = sum([weight[i] * X[i] for i in range(n_X)]) + bias
            # Update weight and bias if prediction is incorrect
            if y * a <= 0:
                weight = [weight[i] + y * X[i] for i in range(n_X)]
                # Update the bias term by adding the label multiplied by the learning rate
                bias += y

    # Return parameters
    return bias, weight

def perceptronTest(bias, weight, X):
    # Compute activation code as the dot product of the weight vector and the input,
    a = sum([weight[i] * X[i] for i in range(len(X))]) + bias
    
    # Return the predicted label based on the sign of the activation code
    if a >= 0:
        return 1
    else:
        return -1

#Question 3

def readData(file_path):
    '''
    readData takes in a file path as input, reads in a CSV
    file located at that path using pandas read_csv function with
    header=None option, and processes the data into a list of tuples.

    the function map class Ys to integer values and creates a list of tuples, where each tuple contains a
    list of the first four columns of the data and an integer value of the
    modified 5th column.
    '''
    df = pd.read_csv(file_path, header=None)
    df[4] = df[4].map({'class-1': 1, 'class-2': 2, 'class-3': 3})
    data = [(df.iloc[i, :4].tolist(), df.iloc[i, 4]) for i in range(len(df))]
    return data

train = readData("train.data")
test = readData("test.data")


def dataPrep(data, classLabel, multiclass=False):
    '''
    The code defines a function called dataPrep
    which takes three arguments: data (a list of tuples),
    class_lab (an integer), and multiclass (a boolean with
    default value of False).
    '''
    # Create an empty list to store the modified data tuples
    newData = []
    # Loop through each tuple in the input data
    for X, Y in data:
        # If multiclass is True, check if the modified class label matches class_lab
        if multiclass:
            if Y == classLabel:
                # If it matches, add the tuple to new_data with a modified label of 1
                newData.append((X, 1))
            else:
                # If it doesn't match, add the tuple to new_data with a modified label of -1
                newData.append((X, -1))
                 # If multiclass is False, only add tuples to new_data if their modified class label doesn't match class_lab
        else:
            if Y != classLabel:
                newData.append((X, Y))
    # Check if new_data is empty
    if not newData:
        # If it is, return an empty list
        return []
    else:
        # Create a new list of tuples to store the final modified data
        new_tuples = []

        # Loop through each tuple in newdata
        for X, Y in newData:
            # If the original label matches the minimum label in new_data, set the modified label to 1, else set it to -1
            new_tuples.append((X, 1 if Y == min([x[1] for x in newData]) else -1))
    # Return the final modified data
    return new_tuples

    
# Accuracy 
def accuracy(train, test, maxIter):
    '''
    this code assumes that there are functions PerceptronTrain and PerceptronTest
    defined elsewhere in the code that take in training data and
    return a weight vector and bias term, and take in a weight vector, bias term,
    and features, and return a predicted class label, respectively.
    '''

    # Separate features and class labels for training data
    X_train = [d[0] for d in train]
    y_train = [d[-1] for d in train]

  # Separate features and class labels for testing data
    X_test = [d[0] for d in test]
    y_test = [d[-1] for d in test]
    
    # Train the perceptron on the training data
    bias, weight = perceptronTrain(train, maxIter)
    
    # Use the trained perceptron to make predictions on the training data
    ytrainpred = [perceptronTest(bias, weight, x) for x in X_train]
    
    # Calculate the numbiaser of correctly classified samples in the training data
    correctTrainsample = sum([1 for i in range(len(y_train)) if y_train[i] == ytrainpred[i]])
    
    # Calculate the accuracy of the perceptron on the training data
    trainAccuracy = round(correctTrainsample / len(y_train) * 100, 2)
    
    # Use the trained perceptron to make predictions on the test data
    ytestpred = [perceptronTest(bias, weight, x) for x in X_test]
    
    # Calculate the numbiaser of correctly classified samples in the test data
    CorrectTestsample = sum([1 for i in range(len(y_test)) if y_test[i] == ytestpred[i]])
    
    # Calculate the accuracy of the perceptron on the test data
    testAccuracy = round(CorrectTestsample / len(y_test) * 100, 2)
    
    # Return a string containing the training and test accuracy
    return f'The is the train accuracy: {trainAccuracy}%\n The is the test accuracy: {testAccuracy}%'


print('Binary Perceptron Class Accuracies')
'''
This code is preparing the data for a multi-class classification problem by converting it into
binary classification problems using the perceptron algorithm. It first prepares the data for
binary classification . Then it  trains the perceptron model, computes and prints 
the accuracy of the model for all the classes.
'''
#Preparing data for class 1 and 2
Binary_class1_2_train = dataPrep(train, 3, multiclass=False)
Binary_class1_2_test = dataPrep(test, 3, multiclass=False)

#Training Perceptron with binary class 1 and 2
bias1_2, weight1_2 = perceptronTrain(Binary_class1_2_train, 20)

#Computing and printing accuracy for binary class 1 and 2
Binary_class1_2_accuracy  = accuracy(Binary_class1_2_train, Binary_class1_2_test, 20)
print(f"For Class 1 and Class 2: \n {Binary_class1_2_accuracy }")

#Preparing data for class 2 and 3
Binary_class2_3_train = dataPrep(train, 1, multiclass=False)
Binary_class2_3_test = dataPrep(test, 1, multiclass=False)

#Training Perceptron with binary class 2 and 3
bias2_3, weight2_3 = perceptronTrain(Binary_class2_3_train, maxIter=20)

#Computing and printing accuracy for binary class 2 and 3
Binary_class2_3_accuracy = accuracy(Binary_class2_3_train, Binary_class2_3_test, 20)
print(f"For Class 2 and Class 3: \n {Binary_class2_3_accuracy}")

#Preparing data for class 1 and 3
Binary_class1_3_train = dataPrep(train, 2, multiclass=False)
Binary_class1_3_test = dataPrep(test, 2, multiclass=False)

#Training Perceptron with binary class 1 and 3
bias1_3, weight1_3 = perceptronTrain(Binary_class1_3_train, 20)

Binary_class1_3_accuracy = accuracy(Binary_class1_3_train, Binary_class1_3_test, 20)
print(f"For Class 1 and Class 3: \n {Binary_class1_3_accuracy}")


#Question 4
#Printing the header for the Multiclass accuracies
print("\n")
print('Multiclass Accuracies')

'''
This code calculates and prints multiclass accuracies using the one-vs-rest approach.

he code uses the dataPrep function to create a new dataset that is binary with the specified
class labeled as 1 and all other classes labeled as -1.

The code then calls the PerceptronTrain function with the binary training dataset and a maximum
number of iterations of 20.

After training, the code uses the accuracy function to calculate the accuracy of the trained
classifier on both the training and testing datasets for the specified binary classification.
'''

# 1 vs rest
# Prepare the training and testing data for classification between Class 1 and the rest
Multi_class1vsrest_train = dataPrep(train, 1, multiclass=True)
Multi_class1vsrest_test = dataPrep(test, 1, multiclass=True)
# Train the perceptron on the training data for classification between Class 1 and the rest
perceptronTrain(Multi_class1vsrest_train, 20)

# Compute and print the accuracy on the training and testing data for classification between Class 1 and the rest
Multi_class1vsrest_accuracy = accuracy(Multi_class1vsrest_train, Multi_class1vsrest_test, 20)
print(f"For Class 1 vs the rest: \n{Multi_class1vsrest_accuracy}")

# 2 vs rest
# Prepare the training and testing data for classification between Class 2 and the rest
Multi_class2vsrest_train = dataPrep(train, 2, multiclass=True)
Multi_class2vsrest_test = dataPrep(test, 2, multiclass=True)
# Train the perceptron on the training data for classification between Class 2 and the rest
perceptronTrain(Multi_class2vsrest_train, 20)

# Compute and print the accuracy on the training and testing data for classification between Class 2 and the rest
Multi_class2vsrest_accuracy = accuracy(Multi_class2vsrest_train, Multi_class2vsrest_test, 20)
print(f"For Class 2 vs the rest: \n{Multi_class2vsrest_accuracy}")

# 3 vs rest
# Prepare the training and testing data for classification between Class 3 and the rest
Multi_class3vsrest_train = dataPrep(train, 3, multiclass=True)
Multi_class3vsrest_test = dataPrep(test, 3, multiclass=True)
# Train the perceptron on the training data for classification between Class 3 and the rest
perceptronTrain(Multi_class3vsrest_train, 20)

# Compute and print the accuracy on the training and testing data for classification between Class 3 and the rest
accuracy_3vsrest = accuracy(Multi_class3vsrest_train, Multi_class3vsrest_test, 20)
print(f"For Class 3 vs the rest: \n{accuracy_3vsrest}")



#Question 5
RC = [0.01, 0.1, 1.0, 10.0, 100.0]

def RegperceptronTrain(train, maxIter, RC):
    # Initialize weighteight vector and bias term
    n_X = len(train[0][0])
    weight = [0.0] * n_X
    ## Weight vector is a list with the same length
    # as the number of features in each training example, initialized with zeros
    bias = 0.0
    # Bias term is initialized with zero


    # Loop over training set for a maximum numbiaser of iterations
    for iter in range(maxIter):
        # Loop over all training examples
        for X, y in train:
            # Compute activation
            a = sum([weight[i] * X[i] for i in range(n_X)]) + bias

            # Update weight and bias if prediction is incorrect
            if y * a <= 0: # Check if the prediction for the current training example is incorrect
                for i in range(n_X): # Update the weights using the current training example and the regularization constant
                    weight[i] = weight[i] + y * X[i] -(2*RC*weight[i])
                    bias += y
    # Return learned parameters
    return bias, weight


def RegAccuracy(train, test, maxIter, RC):
    '''
    This code is implementing a regularized perceptron algorithm for binary classification

    RegPerceptronTrain: This function trains the perceptron model on the input training data
    by updating the weight vector and bias term according to the perceptron update rule, and
    returns the updated weight vector and bias term.

    RegPerceptronTest: This function takes in the trained weight vector and bias term along with
    a new data point and outputs a predicted class label based on the learned decision boundary.

    RegAccuracy: This function evaluates the accuracy of the perceptron model on the training and
    testing data for a range of regularization coefficients. It returns a pandas dataframe containing
    the regularization coefficient and corresponding train and test accuracies.
    '''
    # Initialize an empty list to store the accuracies for each regularization coefficient
    accuracy_list = []

    # Loop through each regularization coefficient in RC
    for coeff in RC:
        # Extract the features and labels from the training dataset
        X_train = [d[0] for d in train]
        y_train = [x[-1] for x in train]

        # Extract the features and labels from the testing dataset
        X_test = [d[0] for d in test]
        y_test = [x[-1] for x in test]

        # Train a perceptron model on the training dataset using the regularization coefficient and maximum number of iterations provided
        bias, weight = RegperceptronTrain(train, maxIter, coeff)

        # Make predictions on the training dataset using the trained model
        ytrainpred = [perceptronTest(bias, weight, x) for x in X_train]

        # Calculate the accuracy of the model's predictions for the training dataset
        correctTrainsample = sum([1 for i in range(len(y_train)) if y_train[i] == ytrainpred[i]])
        trainAccuracy = round(correctTrainsample / len(y_train) * 100, 2)

        # Make predictions on the testing dataset using the trained model
        ytestpred = [perceptronTest(bias, weight, x) for x in X_test]

        # Calculate the accuracy of the model's predictions for the testing dataset
        CorrectTestsample = sum([1 for i in range(len(y_test)) if y_test[i] == ytestpred[i]])
        testAccuracy = round(CorrectTestsample / len(y_test) * 100, 2)

        # Store the regularization coefficient and corresponding accuracies in the accuracy list
        accuracy_list.append([coeff, trainAccuracy, testAccuracy])

    # Convert the accuracy list to a pandas DataFrame and return it
    accuracy_df = pd.DataFrame(accuracy_list, columns=["Regularization Coefficient", "Train Accuracy", "Test Accuracy"]).set_index('Regularization Coefficient')
    return accuracy_df

#Regularised Multiclass
#Print a blank line and a header indicating the following results are for regularized multiclass accuracy

print("\n") 
print('Regularizeation Multiclass Accuracy')
'''
This code is implementing regularized multiclass perceptron algorithm for three classes (class 1, class 2, and class 3).

It loops over the classes, prepares data for each class using the dataPrep() function, and then applies the RegperceptronTrain() 
function to train the regularized perceptron algorithm with a maximum of 20 iterations and a regularization coefficient of 1.0.

Then it computes the accuracy using RegAccuracy() function and prints the accuracy of each class against the rest of the classes.
'''
#Define the classes
classes = [1, 2, 3]

#Loop over each class and perform the following operations
for class_num in classes:
     #Prepare training and testing data for the current class
    train_class = dataPrep(train, class_num, multiclass=True)
    test_class = dataPrep(test, class_num, multiclass=True)
    
    #Train the regularized perceptron on the current class with a regularization constant of 1.0
    RegperceptronTrain(train_class, 20, 1.0)
    
    #Compute and print the accuracy for the current class using the regularized perceptron
    accuracy = RegAccuracy(train_class, test_class, 20, RC)
    print(f"For Class {class_num} vs the rest: \n {accuracy}")