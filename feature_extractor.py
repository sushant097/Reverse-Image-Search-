import numpy as np
from numpy.linalg import norm

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


model = ResNet50(weights='imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3),
                 pooling='max')


def normalize_features(features):
    return features/norm(features)


def extract_features(img_path):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path,
                         target_size=(input_shape[0], input_shape[1]))
    # To np.array. h x w x c. dtype=float32
    img_array = image.img_to_array(img)
    # (224,224,3) -> (1, 224, 224, 3)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(
        expanded_img_array)  # normalize image pixels
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / \
        norm(flattened_features)  # normalize features
    return normalized_features
