from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageOps
import cv2 as cv
import base64
from io import BytesIO
import tensorflow as tf
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from array import array
import operator
import urllib.request
from app import app

ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
# save model with  krtas
model_path = "/Users/apple/Downloads/venv/Flask_SUMMIT_venv/cnn.h5"
tf_config = os.environ.get('TF_CONFIG')
sess = tf.Session(config=tf_config)

# g will get the graph
g = tf.get_default_graph()
global graph
with g.as_default():
   set_session(sess)
   model = tf.keras.models.load_model(model_path)
   model._make_predict_function()
text_file = open("/Users/apple/Downloads/venv/Flask_SUMMIT_venv/cnn_labels.txt", "r")  # just name C:\Users\Prashamsa\Desktop\AppFlask\label.txt
lines = text_file.read().split('\n')
# you have to load the graph
# they dont mention that in the tutorial
# hope this is helpful

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('upload.html')

# @app.route('/', methods=['POST'])
# def upload_file():
#     if request.method == 'POST':
#        # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash('No file selected for uploading')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['APP_ROOT_2'], filename))
#            # print(filename)
#             flash("File uploaded")
#             return render_template('upload.html', filename=filename)
#         else:
#             flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
#             return redirect(request.url, filename=filename)


"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
# @app.route('/predict', methods=['GET', 'POST']) ####################
def pad_and_resize(img, size, rgb=(0,0,0)):
  # Obtains max dimension
  wh = img.size
  max_dim = max(wh)
  # Obtains width and height padding
  width_pad, height_pad = max_dim - wh[0], max_dim - wh[1]
  # Obtains left, right, top, bottom padding
  left_pad, top_pad = width_pad // 2, height_pad // 2
  right_pad, bottom_pad = width_pad - left_pad, height_pad - top_pad
  # Pads the image
  padding = (left_pad, top_pad, right_pad, bottom_pad)
  padded_img = ImageOps.expand(img,padding, 0)
  return padded_img.resize(size, Image.ANTIALIAS)

# Function will determine size of border needed based on
# the area of the bounding box given
def b_size(area):
  border = 0
  if area <= 7000:
    border = 15
  elif area > 7000 and area <= 50000:
    border = 30
  elif area > 50000 and area < 90000:
    border = 50
  else:
    border=75
  return border

"""**************************************************************************"""
def preprocess_image(image_path):
   # input_image = load_encoding(image_path, base64_encoding_str)
   input_image = Image.open(image_path)
   # Preprocess the image first
   im_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # check that f is the image!!!!
   im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
   # Calculate median third
   median_array = image.img_to_array(im_blur)
   median_array = median_array.ravel()
   median = np.median(median_array)
   median_third = median*0.66
   # Threshold the image
   ret, im_th = cv2.threshold(im_gray, median_third, 255, cv2.THRESH_BINARY_INV)
   image, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # Get rectangles contains each contour
   rects = [cv2.boundinRect(ctr) for ctr in ctrs]
   indiv_boxes = []
   count = 0
   for rect in rects:
       number = []
       if rect[2] + rect[3] < 80:
           continue
       # rect object gives tuple with x1, y1, height, width
       count = count + 1

       area = rect[2]*rect[3]
       border = b_size(area)

       # Draw the rectangles
       cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

       # Cropping each bounding box and storing the image inside into roi
       roi = im_th[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]

       # Make number thicker so it becomes easier to recognize
       kernel = np.ones((5,5),np.uint8)
       dilated_im = cv2.dilate(roi, kernel, iterations=1)

       # Making roi into an image so that we can resize
       img_jpg = Image.fromarray(dilated_im)

       # Adding a border to the image
       img_with_border = ImageOps.expand(img_jpg,border=border,fill=0)#15

       # Setting the size for resizing
       size = (28,28)

       # Padding and resizing the image (which is one rectange)
       new_img = pad_and_resize(img_with_border, size=size)

       # Turning the resized image back into an array
       value = img_to_array(new_img)

       # print(list(value))
       y = np.expand_dims(value, axis=0)
##################
       with g.as_default():
           set_session(sess)
           predictions = model.predict(y)
           number.append(np.argmax(predictions))
           number.append(rect)
           #print('prediction: {}'.format(np.argmax(predictions)))
           indiv_boxes.append(number)
   return (indiv_boxes)
"""**************************************************************************"""

@app.route('/predict', methods=['GET', 'POST'])
def crop_and_classify_subimages():
	if request.method == 'POST':
		f = request.files['file']
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

    # Preprocess the image first
	im_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # check that f is the image!!!!
	im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
	# Calculate median third
	median_array = image.img_to_array(im_blur)
	median_array = median_array.ravel()
	median = np.median(median_array)
	median_third = median*0.66
    # Threshold the image
	ret, im_th = cv2.threshold(im_gray, median_third, 255, cv2.THRESH_BINARY_INV)
	image, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Get rectangles contains each contour
	rects = [cv2.boundinRect(ctr) for ctr in ctrs]

	indiv_boxes = []
	count = 0
	for rect in rects:
		number = []
		if rect[2] + rect[3] < 80:
			continue
	    # rect object gives tuple with x1, y1, height, width
		count = count + 1

		area = rect[2]*rect[3]
		border = b_size(area)

	    # Draw the rectangles
		cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

	    # Cropping each bounding box and storing the image inside into roi
		roi = im_th[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])]

	    # Make number thicker so it becomes easier to recognize
		kernel = np.ones((5,5),np.uint8)
		dilated_im = cv2.dilate(roi, kernel, iterations=1)

	    # Making roi into an image so that we can resize
		img_jpg = Image.fromarray(dilated_im)

	    # Adding a border to the image
		img_with_border = ImageOps.expand(img_jpg,border=border,fill=0)#15

	    # Setting the size for resizing
		size = (28,28)

	    # Padding and resizing the image (which is one rectange)
		new_img = pad_and_resize(img_with_border, size=size)

	    # Turning the resized image back into an array
		value = img_to_array(new_img)

	    # print(list(value))
		y = np.expand_dims(value, axis=0)
##################
		with g.as_default():
			set_session(sess)
			predictions = model.predict(y)
			number.append(np.argmax(predictions))
			number.append(rect)
			#print('prediction: {}'.format(np.argmax(predictions)))
			indiv_boxes.append(number)
	return (indiv_boxes)
##################

# Computes shortest eucledian distance from the origin
def distance_from_origin(xi,yi):
    sq1 = xi*xi
    sq2 = yi*yi
    return math.sqrt(sq1 + sq2)

# Orders bounding boxes based on location in the image
def order(predlist):
  rect_to_remove = []
  row_rect_close_to_origin = []
  ordered_list = [] # Output list
  length_of_r_list = len(predlist)

  while len(ordered_list) != length_of_r_list:
    # Default smallest distance from first value in list
    smallest_dist = distance_from_origin(predlist[0][1][0], predlist[0][1][1])
    row_list = []

    # Find boundary box closet to origin
    for rect in predlist:
      dist = distance_from_origin(rect[1][0], rect[1][1])
      if(dist <= smallest_dist):
        smallest_dist = dist
        close_to_origin = rect

        # Get y value of point 2, All boundary boxes with y value
        # of point 1 less than check are added next
        check = close_to_origin[1][1] + close_to_origin[1][3]

    # Append to ordered list
    ordered_list.append(close_to_origin)

    # Remove it from the list
    predlist.remove(close_to_origin)

  # Check if other boundary boxes exist in the same row
    for rect in predlist:
      # Check if pt_1_y is less than pt_2_y
      if(rect[1][1] <= check):
        # Append to a row list
        row_list.append(rect)
    # Remove all bondary boxes from r_list that were in row_list
    [predlist.remove(rect) for rect in row_list]

    print("row_list",row_list)

    # Now check what values in the row are closet to the origin
    for _ in range(len(row_list)):
      # Default smallest distance from first value in list
      smallest_dist = distance_from_origin(row_list[0][1][0], row_list[0][1][1])

      for rect in row_list:
        dist = distance_from_origin(rect[1][0], rect[1][1])

        if(dist <= smallest_dist):
          smallest_dist = dist
          row_rect_close_to_origin = rect
      row_list.remove(row_rect_close_to_origin)

      # Append box closet to origin
      ordered_list.append(row_rect_close_to_origin)

  return ordered_list

# the ordered list of bounding boxes (also contains predictions)
indiv_boxes = crop_and_classify_subimages() ##### should get me indiv boxes
checkList = order(indiv_boxes)
# mathequation will be the array that we turn into str
mathequation = [checkList[i][0] for i in range(len(checkList))]

# Turns the equation array into a string
def to_str(arr):
  operation = ""
  for x in arr:
    if x == 10:
      operation += (str(chr(45)))
    elif x == 11:
      operation += (str(chr(43)))
    elif x == 12:
      operation += (str(chr(42)))
    elif x == 13:
      operation += (str(chr(47)))
    else:
      operation += (str(x))

  return operation

# Save the string version of the equation
new_str = to_str(mathequation)

# Check to make sure the divisions are read in correctly
def right_div(some_str):
  if len(some_str) <= 3:
    return some_str
  for i in range(len(some_str)-3):
    if some_str[i:i+3] == '0-0' and some_str[i+3] != '.':
      some_str = some_str.replace('0-0','/')

  return some_str

new_str = right_div(new_str)
print('The equation is: {}'.format(new_str))

# Get solution and output it
solution = eval(new_str)
print('The solution is: {}'.format(solution))



















"""

   with g.as_default():
       set_session(sess)
       detections = detect_subimages(file_path, detection_model)
   num_detections = detections[0]
   detection_classes = detections[1]
   detection_scores = detections[2]
   detection_boxes = detections[3]
   cropped_image_list = []
   cropped_image_count = 0
   resized_image = preprocess_image(file_path)
   resized_image = image.img_to_array(resized_image)
   resized_height, resized_width = 224, 224
   for i in range(0,int(num_detections[0])):
       left = int(resized_width*detection_boxes[0][i][1])
       top = int(resized_height*detection_boxes[0][i][0])
       right = int(resized_width*detection_boxes[0][i][3])
       bottom = int(resized_height*detection_boxes[0][i][2])
       if detection_scores[0][i] >= 0.5:
           subimage = resized_image[top:bottom, left:right]
           cropped_image_list.append(subimage)
           cropped_image_count += 1
   if cropped_image_count == 0:
       return ("There are no detections, try another image")
   else:
       print("Number of confident detections: {}".format(cropped_image_count))
       predictions = []
       for cropped_image in cropped_image_list:
           resized_subimage = preprocess_subimage(cropped_image)
           resized_subimage = image.img_to_array(resized_subimage)
           with g.as_default():
               set_session(sess)
               prediction = classification_model.predict(np.array([resized_subimage]))
               prediction_class = np.argmax(prediction)
               # predictions.append(prediction)
               predictions.append(classification_labels[prediction_class])
       return (str(predictions))
   return None
"""

"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

#
# @app.route('/show/<filename>')
# def uploaded_file(filename):
#     file = request.files['file']
#     filename = secure_filename(file.filename)
#     # file.save(os.path.join(app.config['APP_ROOT_2'], filename))
#     return render_template("upload.html", filename = filename)
#
# @app.route('/uploads/<filename>')
# def send_file(filename):
#     return send_from_directory(APP_ROOT_2, filename)

if __name__ == "__main__":
    app.run()
