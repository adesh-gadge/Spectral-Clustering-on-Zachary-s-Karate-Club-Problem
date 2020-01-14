import networkx
import pandas as pd
from sklearn.cluster import SpectralClustering

# reading the GML file
G = networkx.read_gml('karate.gml', label='id')
G.nodes()
# converting graph to matrix
df = networkx.to_numpy_matrix(G)
# reading the labels from label.csv
labels=pd.read_csv('labels.csv')

# developing spectral clustering model with k= 2
clustering = SpectralClustering(n_clusters=2,
                                assign_labels="discretize").fit(df)

predicted_label =pd.DataFrame({'Predictions':clustering.labels_})
output_df =pd.concat([labels,predicted_label],axis=1)
# predictions
print(output_df)
