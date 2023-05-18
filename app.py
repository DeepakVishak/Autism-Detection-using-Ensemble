from flask import Flask, render_template, request
import os
from predict import predict


app = Flask(__name__)
from werkzeug.utils import secure_filename



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        f = request.files['fileup']
        f.save('static/files/image.jpg')
        b = predict()
        return render_template("wait.html", abc=b)
    return render_template("home.html")


if __name__ == '__main__':
    app.run(debug=True)
