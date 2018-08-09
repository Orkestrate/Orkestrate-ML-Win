# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def main(imagefolder, modelfolder, modelname):
  image_list = glob.glob(str(imagefolder) + '/*.*')
  model_file = modelfolder + "/retrained_graph.pb"
  label_file = modelfolder + "/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  string = '<style type="text/css">\
    table {\
        border-collapse: collapse;\
        width: 100%;\
    }\
    \
    th, td {\
    	text-align:center;\
    }\
    \
    td:nth-child(even){background-color: #f2f2f2}\
    \
    th {\
        background-color: #3a0345;\
    	color: white;\
    }\
    </style>\
    <center>\
    <table border="1" align="center">\
      <tbody>\
        <tr>\
          <th>Image</th>\
    	  <th>Name</th>\
          <th>Labels</th>\
          <th>Assessment</th>\
        </tr>\
        <tr>'

  for file_name in image_list:

      graph = load_graph(model_file)
      t = read_tensor_from_image_file(file_name,
									  input_height=input_height,
									  input_width=input_width,
									  input_mean=input_mean,
									  input_std=input_std)
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = graph.get_operation_by_name(input_name);
      output_operation = graph.get_operation_by_name(output_name);
      with tf.Session(graph=graph) as sess:
          results = sess.run(output_operation.outputs[0],
    						  {input_operation.outputs[0]: t})
          results = np.squeeze(results)
          top_k = results.argsort()[-5:][::-1]
          labels = load_labels(label_file)
      d, tail = os.path.split(file_name)
      rowspan_no = len(labels)
      rowspan_height = 300
      td_height = rowspan_height / rowspan_no
      string_file_name = '<td width="500" height="' + str(rowspan_height) + '" rowspan="' + str(rowspan_no) + '">\
      <img src="' + file_name.replace('\\','/') + '" height="' + str(rowspan_height-10) + '"></td>\
      <td rowspan="' + str(rowspan_no) + '">' + tail + '</td>'
      string = string + string_file_name
      for i in top_k:
          x = "{:.5f}".format(results[i])
          string_result = '<td height="' + str(td_height) + '">' + str(labels[i]) + '</td>\
                          <td height="' + str(td_height) + '">' + str(x) + '</td>\
                        </tr>\
                        <tr>'
          string = string + string_result
  string_final = string + '</tr>\
                          </tbody>\
                        </table>\
                        </center>'
  return string_final
