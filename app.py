import os
from flask import Flask, request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
# import sys
# import logging
# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template('index.html')
#
# ButtonPressed = 0
# @app.route("/button", methods=['GET', 'POST'])
# def button():
#     if request.method == "POST":
#         ButtonPressed = ButtonPressed + 1
#         return render_template("button.html", ButtonPressed = ButtonPressed)
#     ButtonPressed = ButtonPressed + 1
#     return render_template("button.html", ButtonPressed = ButtonPressed)

# UPLOAD_FOLDER = 'D:/uploads'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
target = os.path.join(APP_ROOT, 'static/')
app.config['APP_ROOT_2'] = target
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
