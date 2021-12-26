import numpy as np
import pickle
from PIL import Image
import uuid
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template


from feature_extractor import extract_features, normalize_features

app = Flask(__name__)

# load saved features
filenames = pickle.load(open('./data/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(
    open('data/features-caltech101-resnet.pickle', 'rb'))
class_ids = pickle.load(open('./data/class_ids-caltech101.pickle', 'rb'))


num_images = len(filenames)
num_features_per_image = len(feature_list[0])
print("Number of images = ", num_images)
print("Number of features per image = ", num_features_per_image)

# Using NearestNeighbors to find similar image
no_of_results_send = 8
neighbors = NearestNeighbors(n_neighbors=no_of_results_send+1,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)
        upload_img_path = 'static/uploaded/' +\
            str(uuid.uuid4())[:8] + '_' + file.filename
        print(upload_img_path)
        img.save(upload_img_path)

        # extract features of query image
        features_extract = extract_features(upload_img_path)

        # Search
        distances, indices = neighbors.kneighbors([features_extract])
        # print(distances)
        # print(indices)

        # Don't take the first closest image as it will be the same image
        # taking first 8 similar images except first image
        name_and_distances = [(distances[0][i], 'static/'+filenames[indices[0][i]])
                              for i in range(1, no_of_results_send+1)]
        # print(name_and_distances)

        return render_template('index.html', query_img_path=upload_img_path, results=name_and_distances)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
