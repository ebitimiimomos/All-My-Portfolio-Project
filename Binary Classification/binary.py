#Step 1- Indicate the imported packages/libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Step-2. Load the dataset and print the data information

'''This code reads in a CSV file using the pandas library's read_csv() function 
Then, it prints the entire DataFrame to the console using the print() function. 
This displays the contents of the dataset, including all rows and columns, 
in a tabular format.'''
dataset = pd.read_csv('dataset_assignment1.csv')
print(dataset) #print dataset


# Step-3. Understand the dataset:
# i. Print out the number of samples for each class in the dataset
'''
This code counts the number of samples in the 'class' column of the dataset and prints the counts for each class.
it  creates a new variable called 'sample' which is a pandas series , then it uses a for loop to iterate over the sample variable and print out the counts for each class. 
Then it uses the 'enumerate' function to keep track of the index of each count, starting at 0.
'''
# Count the number of samples for each class
sample = dataset['class'].value_counts()

# Print the counts for each class and add a sentence
for k, count in enumerate(sample):
    print(f"Class {k+1} has {count} samples.")


#ii. Plot some figures to visualize the dataset - Bar chat and Heatmap
'''
This code creates a set of bar charts for each feature in the dataset, separated by class (0 or 1)
'''

#Bar Chart
#selects the rows in the dataset that belong to class 0 and class 1 and stores them in separate data frames.
class_0_data = dataset.loc[dataset['class'] == 0]
class_1_data = dataset.loc[dataset['class'] == 1]

# Create bar charts for all features, separated by class
# each subplot representing one feature in the dataset. 
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
axs = axs.flatten()

#Iterate over each feature in the dataset, except for the last column (which contains the class label)

for f, feature in enumerate(dataset.columns[:-1]):
    class_0_counts = class_0_data[feature].value_counts()
    class_1_counts = class_1_data[feature].value_counts()
    x_labels = sorted(set(class_0_counts.index) | set(class_1_counts.index))

# Create bar charts for the current feature, separated by class
    axs[f].bar(x_labels, class_0_counts.reindex(x_labels, fill_value=0), color='blue', alpha=0.5, label='Class 0')
    axs[f].bar(x_labels, class_1_counts.reindex(x_labels, fill_value=0), color='orange', alpha=0.5, label='Class 1')
    
    # Set the x-axis label for the current feature
    axs[f].set_xlabel(feature)
    axs[f].legend() # Add a legend to the current subplot

#abels and a title to the plot and shows the plot
plt.suptitle("Bar Charts for Class 0 and Class 1", fontsize=16)
plt.tight_layout()
plt.show()

#Correlation Matrix Heatmap
'''This code generates a correlation matrix heatmap for the features in the dataset. 
It first creates a correlation matrix using the corr() method from pandas.'''
# Create a correlation matrix
corr_matrix = dataset.corr()
# Set up the figure size and style
fig, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=1.2)
# Create a heatmap with the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
# Set the title and axis labels
ax.set_title('Correlation Matrix for Features in Data Set', fontsize=18)
# Rotate the x-axis labels for easier reading
plt.xticks(rotation=45)
plt.show()



# iii. For each class, print-out the statistical description of features
#step 3.3
# Group dataset by class
grouped = dataset.groupby('class') #stores the resulting grouped object
#using a for loop and the variables to store the group's name and data
for name, group in grouped:#Iterates over the group
    print('Statictical Description of features in Class', name)#Print statistical description of features for each class
    print(group.describe().round(2)) #round the statistical values to 2 decimal places.




#Step-4. Split data into a training dataset and a testing dataset.
# Split dataset into two sets. , one for training and one for testing,
X = dataset.drop('class', axis=1) #features variable 
y = dataset['class'] #target variable 
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
spliting the sets is done using the train_test_split() function from scikit-learn. 
the proportion of the data to be used for testing (here, 20% or test_size=0.2).
'''
# Print the size of the training and testing sets
print("The training dataset is:", len(X_train))
print("The testing dataset is:", len(X_test))



# Classification Methods - Random Forest
# 5.1.1 Define hyperparameters in the algorithm

'''
This code performs hyperparameter tuning for a Random Forest Classifier using a grid search and cross-validation scheme.
'''

# Define hyperparameters and cross-validation scheme
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'max_features': ['sqrt', 'log2']}

'''
Contains the number of estimators, the maximum depth of the trees, and the maximum number of features to consider when looking for the best split.
'''

cv = KFold(n_splits=5, shuffle=True, random_state=42)

'''
Defines the cross-validation scheme. In this case, it uses K-fold cross-validation with 5 splits.
'''

# Perform grid search with Random Forest Classifier
grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'], refit='accuracy')
grid_search.fit(X_train, y_train)

'''
Performs a grid search over the hyperparameters and returns the best hyperparameters and evaluation metrics.
Fits the Random Forest Classifier with different hyperparameters combinations and evaluates it using cross-validation.
'''

# Returns the estimator with the best hyperparameters
best_estimator = grid_search.best_estimator_

# Print the best hyperparameters and evaluation metrics
best = grid_search.best_index_
print(f"The Best parameters for the Random Forest Model is: {grid_search.best_params_}\nThe Random Forest Best accuracy: {grid_search.best_score_}\nThe Random Forest Best precision: {grid_search.cv_results_['mean_test_precision'][best]}\nThe Random Forest Best recall: {grid_search.cv_results_['mean_test_recall'][best]}\nThe Random Forest Best f1-score: {grid_search.cv_results_['mean_test_f1'][best]}")

'''
Prints the best hyperparameters and the evaluation metrics, including accuracy, precision, recall, and f1-score, obtained using cross-validation.
'''

# 5.1.2. Evaluation metrics
# Evaluate the performance of a trained Random Forest classifier model on both the training and test sets.

# Predict target variable for training set
y_pred_train = best_estimator.predict(X_train)

# Print classification report using the classification report for the training set
print('The Random Forest Classification Report (Training Set):\n', classification_report(y_train, y_pred_train))

# Predict target variable for test set
y_pred = best_estimator.predict(X_test)

# Print classification report using the classification report for the testing set
print('The Random Forest Classification Report (Test Set):\n', classification_report(y_test, y_pred))

# Create confusion matrix for training and test set
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn's heatmap function
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

# Add labels, title and axis ticks to the plot
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Random Forest Confusion Matrix (Test Set)')
plt.show()



#plot line grapgh for visualization
'''
evaluating the performance of a Random Forest Classifier model on the test set for different values of the 
 hyperparameter. 
'''
# Define the range of n_estimators values to plot
n_values = [10, 50, 100, 200]
# Define lists to store the evaluation scores for each n_estimators value
test_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

# Train a Random Forest model with each n_estimators value and record the evaluation scores
#loop to train several Random Forest models with different values 
for n in n_values:
    rf_model = RandomForestClassifier(n_estimators=n)
    rf_model.fit(X_train, y_train)
    y_pred_test = rf_model.predict(X_test)
    test_scores['accuracy'].append(accuracy_score(y_test, y_pred_test))
    test_scores['f1'].append(f1_score(y_test, y_pred_test))
    test_scores['precision'].append(precision_score(y_test, y_pred_test))
    test_scores['recall'].append(recall_score(y_test, y_pred_test))

# Create four line graphs, one for each metric
for metric in ['accuracy', 'f1', 'precision', 'recall']:
    plt.plot(n_values, test_scores[metric], label=metric.capitalize())
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel(metric.capitalize())
    plt.title('Random Forest Test Metrics')
    plt.legend()
plt.show()





# Logistic Regression
#5.1.1 Define hyperparameters in the algorithm
'''
This code performs a grid search with logistic regression to find the best hyperparameters and evaluation metrics. 
'''
# Define hyperparameters and cross-validation scheme
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
#hyperparameters being searched are the regularization strength parameter C, with a range of values from 0.001 to 100
cv = KFold(n_splits=5, shuffle=True, random_state=42)
#used is KFold with 5 splits
# Perform grid search with Logistic Regression
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid=param_grid, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'], refit='accuracy', error_score='raise').fit(X_train, y_train)
# Print the best hyperparameters and evaluation metrics
best = grid_search.best_index_
print(f"The Best parameters for logistic regression model: {grid_search.best_params_}\nLogistic regression Best accuracy: {grid_search.best_score_}\nLogistic regression Best precision: {grid_search.cv_results_['mean_test_precision'][best]}\nLogistic regression Best recall: {grid_search.cv_results_['mean_test_recall'][best]}\nLogistic regression Best f1-score: {grid_search.cv_results_['mean_test_f1'][best]}")


#5.1.2. Evaluation metrics for Logistic Regression
#evaluating a Logistic Regression model on a binary classification problem
# fitting a logistic regression model 
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

best_lr_model = lr_model

# Predict target variable for training set
y_pred_train = best_lr_model.predict(X_train)

# Print classification report for training set
print('Logistic Regression Classification Report (Training Set):\n', classification_report(y_train, y_pred_train))

# Predict target variable for test set
y_pred = best_lr_model.predict(X_test)

# Print classification report for test set
print('Logistic Regression Classification Report (Test Set):\n', classification_report(y_test, y_pred))

# Create confusion matrix for training and Test set
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn's heatmap function
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

# Add labels, title and axis ticks to the plot
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Logistic Regression Confusion Matrix (Test Set)')
plt.show()


#plot line graph for visualization
# Define lists to store the evaluation scores for each C value
test_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

# Define the range of C values to try
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Train a logistic regression model with each C value and record the evaluation scores
for C in C_values:
    lr_model = LogisticRegression(C=C)
    lr_model.fit(X_train, y_train)
    y_pred_test = lr_model.predict(X_test)
    test_scores['accuracy'].append(accuracy_score(y_test, y_pred_test))
    test_scores['f1'].append(f1_score(y_test, y_pred_test))
    test_scores['precision'].append(precision_score(y_test, y_pred_test))
    test_scores['recall'].append(recall_score(y_test, y_pred_test))

# Create four line graphs, one for each metric
for metric in ['accuracy', 'f1', 'precision', 'recall']:
    plt.plot(C_values, test_scores[metric], label=metric.capitalize()) #reates a line graph for each metric
    plt.xscale('log')
    plt.xlabel('C value')
    plt.ylabel(metric.capitalize()) #evaluation score for each metric
    plt.title('Logistic Regression Test Metrics')
    plt.legend()
plt.show() #display the graph.



#KNN
#5.1.1 Define hyperparameters in the algorithm

'''
This code performs a grid search with cross-validation to find the best hyperparameters for a KNN classifier. 
'''
# Define hyperparameters and cross-validation scheme
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
#uses a range of k values for n_neighbors

cv = KFold(n_splits=5, shuffle=True, random_state=42)
#evaluates the model using 5-fold cross-validation.

# Perform grid search with KNN
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1'], refit='accuracy', error_score='raise').fit(X_train, y_train)

# Set best_knn_model to the best estimator found by GridSearchCV
best_knn_model = grid_search.best_estimator_

# Print the best hyperparameters and evaluation metrics
best = grid_search.best_index_
print(f"The Best parameters for KNN: {grid_search.best_params_}\nKNN Best accuracy: {grid_search.best_score_}\nKNN Best precision: {grid_search.cv_results_['mean_test_precision'][best]}\nKNN Best recall: {grid_search.cv_results_['mean_test_recall'][best]}\nKNN Best f1-score: {grid_search.cv_results_['mean_test_f1'][best]}")


#5.1.2. Evaluation metrics
'''evaluating the performance of the best KNN model on the training and test sets.'''

# Predict target variable for training set ussing the predict method of the best_knn_model object,
y_pred_train = best_knn_model.predict(X_train)

# Print classification report for training set using the classification_report function from scikit-learn's metrics module.
print('Classification Report (Training Set):\n', classification_report(y_train, y_pred_train))

# Predict target variable for testing set ussing the predict method of the best_knn_model object,
y_pred = best_knn_model.predict(X_test)

# Print classification report for test set using the classification_report function from scikit-learn's metrics module.
print('Classification Report (Test Set):\n', classification_report(y_test, y_pred))

# Create confusion matrix for test set using the confusion_matrix function from scikit-learn's metrics module
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn's heatmap function
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

# Add labels, title and axis ticks to the plot
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('KNN Confusion Matrix (Test Set)')
plt.show()





#Plot graph for visualization
'''
This code performs a hyperparameter tuning for the KNN algorithm by testing different values of the number of neighbors, k.
'''

# Define the range of k values to plot
k_values = range(1, 30)

# Define a dictionary to store the test scores for each metric
test_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

# Train a KNN model with each k value and record the evaluation scores
for k in k_values:
    # iterates over each value of k and trains a KNN model with that number of neighbors.
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train) #trains the model using the training data. 
    y_pred_test = knn_model.predict(X_test) #make predictions on the test data
    test_scores['accuracy'].append(accuracy_score(y_test, y_pred_test))
    test_scores['f1'].append(f1_score(y_test, y_pred_test))
    test_scores['precision'].append(precision_score(y_test, y_pred_test))
    test_scores['recall'].append(recall_score(y_test, y_pred_test))

'''
The evaluation scores for each metric (accuracy, F1 score, precision, and recall) are computed using the corresponding scikit-learn 
unctions (accuracy_score(), f1_score(), precision_score(), and recall_score()) and appended to the test_scores dictionary.
'''
# Create a line graph to show the test scores for each metric
# plot the test scores for each metric against the number of neighbors. 
plt.plot(k_values, test_scores['accuracy'], label='Accuracy')
plt.plot(k_values, test_scores['f1'], label='F1 Score')
plt.plot(k_values, test_scores['precision'], label='Precision')
plt.plot(k_values, test_scores['recall'], label='Recall')

#add labels and a title to the graph.
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Test Evaluation Scores')
plt.legend() #used to add a legend to the graph
plt.show()






