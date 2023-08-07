#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(97) # set to reproducibility of results

#loads the dataset into a Pandas DataFrame using the read_csv method
data = pd.read_csv('dataset', delimiter=' ', header=None) #no header row

# extracts the numerical features from the DataFrame and stores them in an array X
X = data.iloc[:, 1:].astype(float).values

#calculates the Euclidean distance between two vectors, X and Y
def distance(X, Y):
    # Returns the calculated the square root of the sum of the squares of the differences 
    # between the corresponding elements.
    return np.linalg.norm(X - Y)


def silhouetteCoeff(data, clusters):
    '''
    To not distrupt the flow of the answers and to make the code easier to read, The silhouette coefficient has been placed here.
    This silhouette coefficient funtion computes  the silhouette coefficient for the given clustering of data points within the code. 
    The function computes the average distance between all the data and selects the minimum of these distances, Finally, it computes the silhouette coefficient  and takes the mean across all data points.

    '''    
    # Get the number of data points
    n = len(data)
    # Initialize arrays to store distances within and between clusters for each data point
    a = np.zeros(n)
    b = np.zeros(n)
    # For each data point, compute the average distance to all other data points in its own cluster
    for i in range(n):
        a[i] = np.mean(np.apply_along_axis(distance, 1, data[clusters == clusters[i]], data[i]))
        # Compute the average distance to all data points in the nearest neighboring cluster
        other_clusters = np.unique(clusters[clusters != clusters[i]])
        b[i] = np.min([np.mean(np.apply_along_axis(distance, 1, data[clusters == j], data[i]))
                       for j in other_clusters])
    # Compute the silhouette coefficient for each data point
    s = (b - a) / np.maximum(a, b)
    # Return the mean silhouette coefficient across all data points
    return np.mean(s)


def plot_silhouette_coefficients(k_value, silhouette_Coeff, title): # Plot the silhouette coefficients for varying k values.
    # plot line graph for the silhouette coefficients and k values
    plt.plot(k_value, silhouette_Coeff, marker='o', markersize=5, color='red')
    plt.xlabel('k_value') # set the x-axis label to 'k_value'
    plt.ylabel('Silhouette coefficient') # set the y-axis label to 'Silhouette coefficient'
    plt.title(title) # set the plot title to the value of the 'title' parameter
    plt.tight_layout()
    plt.show() # display the plot
   
    
# Define the range of k values to use
k_value = range(2, 10)



# QUESTION 1 -  Implement the k-means clustering algorithm 
'''
This is the implementation of the k-means clustering algorithm. The function uses numpy to efficiently cluster the input data into k clusters, 
where k is a parameter passed to the function. The output of the function is the final cluster assignments and the final centroids.
'''
def kMeans(data, k):
    # Initialize centroids randomly
    initial_centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]
    while True:
        # Assign each data point to the closest centroid
        distances = np.sqrt(np.sum((data[:, np.newaxis] - initial_centroids)**2, axis=2))
        # Compute the Euclidean distance between each data point and each centroid
        # and assign the data point to the closest centroid
        cluster_assignments = np.argmin(distances, axis=1)
        # Find the index of the closest centroid for each data point
        # Compute new centroids based on the mean of the data points in each cluster
        new_centroids = np.array([data[cluster_assignments == i].mean(axis=0) for i in range(k)])
        # Calculate the mean of the data points that belong to each cluster and set the new centroids
        if np.array_equal(new_centroids, initial_centroids):
            # If the new centroids are the same as the old centroids, the algorithm has converged
            break
        
        initial_centroids = new_centroids
        # Update the centroids to the new centroids for the next iteration

    return cluster_assignments, new_centroids
    # Return the final cluster assignments and centroids


# QUESTION 2-  Implement the k-means++ clustering algorithm
'''
This is the implementation of the K-means++ initialization algorithm. It is a modified version of the K-means algorithm that selects the 
initial centroids in a way that improves the convergence rate and reduces the sensitivity to the initial conditions. The K-means 
clustering algorithm is then applied using the initial centroids obtained from the K-means++ algorithm to assign data points and 
returns the optimized centroids and the cluster assignments for each data point.
'''
def kMeansPlusPlus(data, k):
    # Create an empty array of size (k, number of features in data)
    initial_centroids = np.zeros((k, data.shape[1]))
    # Choose a random data point to be the first centroid
    initial_centroids[0] = data[np.random.choice(data.shape[0], 1), :]

# Select the remaining centroids using the K-means++ initialization method
    for i in range(1, k):
        # Calculate the distance between each data point and the existing centroids
        distances = np.zeros((data.shape[0], i))
        for j in range(i):
            distances[:, j] = np.linalg.norm(data - initial_centroids[j], axis=1)
        min_dists = np.min(distances, axis=1)
        # Compute the probability of choosing a data point as the next centroid based on the distance
        probs = min_dists / np.sum(min_dists)
        # Randomly select the next centroid based on the probability distribution
        initial_centroids[i] = data[np.random.choice(data.shape[0], 1, p=probs), :]
    # Use the k-means algorithm to assign data points to the nearest centroid and optimize centroids
    cluster_assignments, centroids = kMeans(data, k)

    return centroids, cluster_assignments
    # Return the final cluster assignments and centroids


# QUESTION 3 - Bisecting k-Means hierarchical clustering algorithm
'''
    This is the implementation of the Bisecting k-Means hierarchical algorithm. The algorithm starts with all data points in a single 
    cluster, and then repeatedly bisects the cluster with the highest sum of squared errors (SSE) using k-means clustering. The function 
    takes two arguments, data and k. The function returns a list of tuples representing the hierarchy of clusters, 
    where each tuple contains the current cluster assignments and centroids.
'''
def bisectingkMeans(data, k):
    cluster_assignments = np.zeros(data.shape[0], dtype=int) # Initialize cluster assignments to all zeros
    initial_centroids = [data.mean(axis=0)] # Set initial centroids to the mean of the data
    hierarchy = [(cluster_assignments, initial_centroids)] # Initialize hierarchy with the first cluster assignments and initial centroids
   
    # Continue bisecting until the desired number of clusters is reached
    while len(hierarchy) < k:
        # Calculate SSE for each cluste
        sse = [np.sum([distance(x, c) ** 2 for x in data[cluster_assignments == i]])
               for i, c in enumerate(initial_centroids)]
        
        # Find the index of the cluster with the largest SSE
        largestClusterSSE = np.argmax(sse) 
        # Split the largest cluster using k-means clustering
        clusterData = data[cluster_assignments == largestClusterSSE]
        new_assignments, centroids = kMeans(clusterData,2)
        # Reassign cluster assignments based on k-means output
        new_assignments[new_assignments == 0] = largestClusterSSE
        new_assignments[new_assignments == 1] = len(initial_centroids)
        # Update cluster assignments and centroids in the hierarchy
        cluster_assignments[cluster_assignments == largestClusterSSE] = new_assignments
        initial_centroids.pop(largestClusterSSE)
        initial_centroids += list(centroids)
        hierarchy.append((cluster_assignments.copy(), initial_centroids.copy()))
       
    return hierarchy
    #Return Hierarchy


# QUESTION 4 - Run, Compute and  k-means clustering algorithm 
kMeans_silhouette_Coeff = []
'''
This determines the opptimal number of clusters to use for clustering a given dataset using the silhouette score metric. 
It also create an empty list to hold the silhouette coefficients for each value of k.

'''
# Loop through each value of k in k_value
for k in k_value:
    # Perform k-means clustering on the dataset X with k clusters and get the resulting clusters and centroids
    clusters_kMeans, centroids_kMeans = kMeans(X, k)
    # Calulate the silhouette coefficient for the resulting clusters and append it to the kMeans_silhouette_Coeff list
    kMeans_silhouette_Coeff.append(silhouetteCoeff(X, clusters_kMeans))

# Print out the silhouette coefficients for each value of k using the k-means algorithm
print("\nK means Algorithm:") #print heading

# Print the k value and Silhouette coefficient
for i, SC in enumerate(kMeans_silhouette_Coeff):
    k = k_value[i]
    print(f"The Silhouette coefficient for the k value of {k}: {SC}")

# Plot the silhouette coefficients for the k-means algorithm using the plot_silhouette_coefficients function
plot_silhouette_coefficients(k_value, kMeans_silhouette_Coeff, 'k Means Graph')

# Determine the optimal value of k based on the highest silhouette coefficient
optimal_k = k_value[np.argmax(kMeans_silhouette_Coeff)]
# Print the optimal value of k to the console
print(f"The optimal number of clusters based on the silhouette score metric is: {optimal_k}")



#QUESTION 5- Run, Compute and  k-means++ clustering algorithm 
'''
This determines the opptimal number of clusters to use for clustering a given dataset using the silhouette score metric. 
It also create an empty list to hold the silhouette coefficients for each value of k.

'''
# Create an empty list to hold the silhouette coefficients for each value of k.
kMeansPlusPlus_silhouette_Coeff = []

# Iterate over each k value and compute the Silhouette coefficient using both kMeans and kMeansPlusPlus
for k in k_value:
    # Compute the clusters and centroids using kMeansPlusPlus
    centroids_kMeansPlusPlus, clusters_kMeansPlusPlus = kMeansPlusPlus(X, k)
    # Compute the Silhouette coefficient for these clusters
    kMeansPlusPlus_silhouette_Coeff.append(silhouetteCoeff(X, clusters_kMeansPlusPlus))

print('\nK-means++ Algorithm:') #print heading

# Print the k value and Silhouette coefficient
for i, SC in enumerate(kMeansPlusPlus_silhouette_Coeff):
    k = k_value[i]
    print(f"The Silhouette coefficient for the k value of {k}: {SC}")

# Plot the Silhouette coefficients for kMeansPlusPlus
plot_silhouette_coefficients(k_value, kMeansPlusPlus_silhouette_Coeff, 'k-Means++ Graph')

# Find the optimal number of clusters for the given data set, based on the silhouette score metric
optimal_k = k_value[kMeansPlusPlus_silhouette_Coeff.index(max(kMeansPlusPlus_silhouette_Coeff))]
print(f"The optimal number of clusters based on the silhouette score metric is {optimal_k}.")




# QUESTION 6 - Run, Compute and Bisecting k-means clustering algorithm 
'''
This calculates the silhouette coefficients for each level of the bisecting k-means clustering hierarchy, 
and also finds the optimal number of clusters based on the silhouette score metric.
'''

# Calculate the hierarchy of clusterings
hierarchy = bisectingkMeans(X, k=9)

# Calculate the Silhouette coefficients for each level of the hierarchy
bisect_sil_coeff = [silhouetteCoeff(X, hierarchy[k-1][0]) for k in k_value]

print('\nBisecting k-Means Algorithm:') #print heading

# Print the k value and Silhouette coefficient
for i, SC in enumerate(bisect_sil_coeff):
    k = k_value[i]
    print(f"The Silhouette coefficient for the k value of {k}: {SC}")

# Plot the k value and Silhouette coefficient
plot_silhouette_coefficients(k_value, bisect_sil_coeff, 'Bisecting k-Means Graph')

# Determine the optimal number of clusters based on the silhouette score metric
optimal_k = k_value[bisect_sil_coeff.index(max(bisect_sil_coeff))]
print(f"\nThe optimal number of clusters based on the silhouette score metric: {optimal_k}")
