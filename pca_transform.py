from sklearn.decomposition import PCA



def transform_features(feature_list, num_feature_dimensions=100):
    "Return: compressed_features"
    pca = PCA(n_components=num_feature_dimensions)
    pca.fit(feature_list)
    return pca.transform(feature_list)
