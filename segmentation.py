import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("customers.csv")

X = data[['Annual_Spend', 'Visit_Frequency']]

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

data.to_csv("segmented_customers.csv", index=False)

plt.scatter(data['Annual_Spend'], data['Visit_Frequency'], c=data['Cluster'])
plt.xlabel("Annual Spend")
plt.ylabel("Visit Frequency")
plt.title("Customer Segments")
plt.show()
