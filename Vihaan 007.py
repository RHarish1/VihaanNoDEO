import tensorflow as tf
import numpy as np
import pandas as pd

# Load data from CSV
data = pd.read_csv('tourism_data.csv')

# Extract features (tags) from the data
tags = data.iloc[:, 1:].values  # Assuming tags are in columns 2 and onwards

# Parameters
num_clusters = 2
num_iterations = 100

# Convert numpy array to TensorFlow constant
tags_tf = tf.constant(tags, dtype=tf.float32)

# Initialize centroids randomly
centroids = tf.Variable(tf.slice(tf.random.shuffle(tags_tf), [0, 0], [num_clusters, -1]))

# Model
for _ in range(num_iterations):
    # Assign each point to the nearest centroid
    expanded_vectors = tf.expand_dims(tags_tf, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)
    
    # Update centroids
    updates = []
    for c in range(num_clusters):
        cluster_points = tf.gather(tags_tf, tf.where(tf.equal(assignments, c)))
        centroid = tf.reduce_mean(cluster_points, axis=[0])
        updates.append(centroid)
    centroids = tf.concat(updates, 0)

# Find the most relevant places based on input tags
input_tags = [1, 0, 1, 0]  # Example input tags
input_tags_tf = tf.constant(input_tags, dtype=tf.float32)
input_tags_expanded = tf.expand_dims(input_tags_tf, 0)
input_tags_expanded = tf.tile