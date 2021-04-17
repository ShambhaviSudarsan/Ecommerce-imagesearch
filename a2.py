from flask import request, Flask
import numpy as np
from numpy.linalg import norm
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

from flask_cors import CORS

app = Flask(__name__)

CORS(app)

feature_vector_list = []

@app.route("/extract_features", methods = ['GET','POST'])
def extract_features():
	print("Before")
	data = request.get_json(force = True)
	print("Data Arrives",data)
	data_values = list(data.values())
	product_id = data_values[0]
	image_path = data_values[1]
	
	model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

	def extract_features(img_path, model):
		input_shape = (128,128,3)
		
		img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
		img_array = image.img_to_array(img)
		
		expanded_img_array = np.expand_dims(img_array, axis=0)
		preprocessed_img = preprocess_input(expanded_img_array)
		
		features = model.predict(preprocessed_img)
		
		flattened_features = features.flatten()
		normalized_features = flattened_features/norm(flattened_features)
		
		return normalized_features

	features = extract_features(image_path, model)
	feature_vector_list.append(features)

	return {"product_id" : product_id, "image_feature" : features.tolist()}

@app.route("/image_search", methods = ['GET','POST'])
def image_search():
	data = request.get_json(force = True)
	data_values = list(data.values())
	image_path = data_values[0]
	
	model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

	def extract_features(img_path, model):
		input_shape = (128,128,3)
		
		img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
		img_array = image.img_to_array(img)
		
		expanded_img_array = np.expand_dims(img_array, axis=0)
		preprocessed_img = preprocess_input(expanded_img_array)
		
		features = model.predict(preprocessed_img)
		
		flattened_features = features.flatten()
		normalized_features = flattened_features/norm(flattened_features)
		
		return normalized_features

	features = extract_features(image_path, model)

	neighbors = NearestNeighbors(n_neighbors = 5, algorithm='brute', metric='euclidean').fit(feature_vector_list)
	indices = neighbors.kneighbors([query_image_features])

	return indices

if __name__ == "__main__":
	app.run()
