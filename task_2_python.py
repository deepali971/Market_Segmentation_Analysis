import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.utils import resample
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


print("\n1. Converting Ordinal Data to Numeric")
data = {'Like': ["I love it!+5", "+4", "+3", "0", "-1", "-2", "-3", "-4", "I hate it!-5"]}
df = pd.DataFrame(data)
like_mapping = {"I love it!+5": 5, "+4": 4, "+3": 3, "0": 0, "-1": -1, "-2": -2, "-3": -3, "-4": -4, "I hate it!-5": -5}
df['Like_n'] = df['Like'].map(like_mapping)
print(df)


print("\n2. Creating Regression Formula Dynamically")
independent_vars = ["yummy", "convenient", "spicy", "fattening", "greasy",
                    "fast", "cheap", "tasty", "expensive", "healthy", "disgusting"]
dependent_var = "Like_n"
formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
print("Regression Formula:", formula)


print("\n3. Mixtures of Regression Models (Latent Class Regression)")
np.random.seed(1234)
X = np.random.rand(100, len(independent_vars))
y = np.random.randint(-5, 6, 100)
gmm = GaussianMixture(n_components=2, random_state=1234)
gmm.fit(X)
clusters = gmm.predict(X)
print("Cluster Assignments:", clusters[:10])
print("Cluster Means:\n", gmm.means_)


print("\n4. Data Transformation: Convert Yes/No to Numeric")
df_yes_no = pd.DataFrame({'feature1': ['Yes', 'No', 'Yes', 'No', 'Yes'],
                          'feature2': ['No', 'Yes', 'Yes', 'No', 'No'],
                          'feature3': ['Yes', 'Yes', 'No', 'No', 'Yes']})
df_numeric = df_yes_no.apply(lambda col: col.map(lambda x: 1 if x == "Yes" else 0))
print(df_numeric)


#print("\n5. PCA Explained Variance")
pca = PCA()
X_pca = pca.fit_transform(X)
print("\n5. PCA Explained Variance:", pca.explained_variance_ratio_.round(4))


print("\n6. k-Means Clustering for Market Segmentation")
distortions = [KMeans(n_clusters=k, random_state=1234, n_init=10).fit(X).inertia_ for k in range(2, 9)]
plt.figure()
plt.plot(range(2, 9), distortions, marker='o')
plt.title("Scree Plot - k-Means Clustering")
plt.show()


print("\n7. Hierarchical Clustering")
Z = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()


print("\n8. Bootstrapped Clustering Stability Analysis")
stability_scores = [np.mean([adjusted_rand_score(
    KMeans(n_clusters=k, random_state=1234, n_init=10).fit(resample(X, y, random_state=1234)[0]).labels_, 
    KMeans(n_clusters=k, random_state=1234).fit(X).labels_) 
    for _ in range(100)]) for k in range(2, 9)]
plt.figure()
plt.plot(range(2, 9), stability_scores, marker='o')
plt.title("Global Stability Analysis")
plt.show()


print("\n9. Two-Step Clustering")
agglo_cluster = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
print(pd.Series(agglo_cluster).value_counts().rename("count"))


print("\n10. Evaluating Segmentation Using a Decision Matrix")
silhouette_scores = [silhouette_score(X, KMeans(n_clusters=k, random_state=1234).fit_predict(X)) for k in range(2, 9)]
plt.figure()
plt.plot(range(2, 9), silhouette_scores, marker="o")
plt.title("Segment Stability Evaluation")
plt.show()


print("\n11. Bagged Clustering")
bagging_kmeans = BaggingClassifier(estimator=KMeans(n_clusters=10), n_estimators=50, random_state=1234).fit(X, y)
plt.figure()
plt.hist(bagging_kmeans.predict(X), bins=np.arange(11)-0.5, edgecolor="black", alpha=0.7)
plt.title("Bagged Clustering Histogram")
plt.show()

print("\n12. Distance-Based Clustering")
risk_dist = squareform(pdist(X, metric="cityblock"))
Z = linkage(risk_dist, method="complete")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering - Manhattan Distance")
plt.show()


print("\n13. k-Means Clustering with Mixture Models")
gmm = GaussianMixture(n_components=4, random_state=1234).fit(X)
print("Cluster Assignments (Mixture Model):", np.bincount(gmm.predict(X)))
print("Log Likelihood:", gmm.score(X))


print("\n14. Mixture of Normal Distributions")
subset_data = X[:, :2]
counts = np.apply_along_axis(lambda x: np.bincount(x.astype(int), minlength=2), axis=0, arr=subset_data)
print("Segment Counts:\n", counts)
print("Segment Probabilities:", np.round(counts.prod() * 100, 2))


print("\n15. Market Segmentation on Binary Data")
binary_data = (X > np.median(X)).astype(int)
counts = binary_data.mean(axis=0)
print("Feature Probabilities:", np.round(counts, 2))
print("Overall Probability:", np.round(np.prod(counts) * 100, 2))
