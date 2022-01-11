import os
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential, layers

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./upload"
app.config['MAX-CONTENT-PATH'] = 2000 * 1000 * 1000
app.config['SECRET_KEY'] = 'experiments'

MnistModel = load_model("model/model.h5")
class_names = ['t-shirt/top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
probability_model = Sequential([MnistModel, layers.Softmax()])

@app.route("/")
def homepage():
	return render_template("home.html")

@app.route("/fashionMnist")
def fashionMnist():
	return render_template("fashionMnist.html")

@app.route("/detectMnistFashion", methods=["GET","POST"])
def detectMnistFashion():
	if request.method == "POST":
		file = request.files["file"]
		if file.filename == "":
			return "no file detected, <br><a href=\"/fashionMnist\">back to fashionMnist</a>"

		filename = secure_filename(file.filename)
		file.save("upload/" + filename)
		tf_image = image.load_img("upload/" + filename,
					color_mode="grayscale",
					target_size=(28,28),
					interpolation="nearest")
		np_image = image.img_to_array(tf_image)
		pred_img = 1 - np.reshape(np_image, (1,28,28,1)) / 250
		predictions = probability_model.predict(pred_img)
		nameFashion = class_names[np.argmax(predictions[0])]
		percentPredict = "{:.2f}%".format(100 * np.max(predictions))
		return "{}, predict: {}<br><a href=\"/fashionMnist\">back to fashionMnist</a>".format(nameFashion, percentPredict)

	return "please submit your image <br><a href=\"/fashionMnist\">back to fashionMnist</a>"


@app.route("/dogAndCat")
def dogAndCat():
	return "dog And Cat"


if __name__ == "__main__":
	app.run(threaded=True)