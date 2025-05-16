import pandas as pd
from pandasgui import show
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from classifier import Classifier

data = load_breast_cancer()
target_class = data.target

df = pd.DataFrame(data=data.data, columns=data.feature_names)
show(df)

# Before Using Unsupervised Learning
plt.title("Mean Radius vs Mean Texture - Before Clustering")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.scatter(df['mean radius'], df['mean texture'])
plt.savefig('./exports/scatter_plot_before.png')
plt.show()

# Elbow Technique
n_k = range(1, 10)
SSE = [] # Sum of Square Error
for k in n_k:
    k_means = KMeans(n_clusters=k)
    k_means.fit(df[['mean radius', 'mean texture']])
    SSE.append(k_means.inertia_) # inertia gives the sum of square error

plt.title("Sum of Square Error")
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(SSE, marker='o')
plt.savefig('./exports/SSE.png')
plt.show()

# After using Unsupervised Learning
k_means = KMeans(n_clusters=2)
df['cluster'] = k_means.fit_predict(df[['mean radius', 'mean texture']])

cluster_1 = df[df.cluster == 0] 
cluster_2 = df[df.cluster == 1]

plt.title("Mean Radius vs Mean Texture - After Clustering")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.scatter(cluster_1['mean radius'], cluster_1['mean texture'])
plt.scatter(cluster_2['mean radius'], cluster_2['mean texture'])
plt.savefig('./exports/scatter_plot_after.png')
plt.show()

# Accuracy
acc_1 = accuracy_score(target_class, df['cluster'])
acc_2 = accuracy_score(target_class, 1 - df['cluster']) 

# We also revert the labels since unsupersived algorithms doesn't have information
# about real classes. He simply regroup the data between two clusters
# an example would be:
# 
# labels = [0, 0, 1, 1, 1, 0, 1]
# target = [1, 1, 0, 0, 0, 1, 0]  # real class
#
# if we compare the labels with the target directly, it seems everything is wrong
# but if we invert the label of the cluster they match perfectly with accuracy 100%
#
# 1 - labels = [1, 1, 0, 0, 0, 1, 0]

accuracy = max(acc_1, acc_2)
print("Clusters match with the target class: ", f"{accuracy * 100}%")

# DecisionTreeClassifier
classifier = Classifier(df[['mean radius', 'mean texture']], target_class)
classifier.train()
print("DecisionTreeClassifier match with the target class: ", f"{classifier.predict() * 100}%")