#!/usr/bin/env python3.8

from flask import Flask, render_template, request
from onnx_modifier import onnxModifier
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/open_model', methods=['POST'])
def open_model():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    onnx_file = request.files['file']

    global onnx_modifier
    onnx_modifier = onnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)

    return 'OK', 200


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    modify_info = request.get_json()
    # print(modify_info)

    onnx_modifier.reload()   # allow downloading for multiple times

    onnx_modifier.modify(modify_info)
    onnx_modifier.check_and_save_model()

    return 'OK', 200

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0")
