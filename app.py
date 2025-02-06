import argparse
import logging
import time
from flask import Flask, render_template, request, send_file
from onnx_modifier import onnxModifier
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
onnx_modifier = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open_model', methods=['POST'])
def open_model():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    onnx_file = request.files['file']

    global onnx_modifier
    onnx_modifier = onnxModifier.from_name_protobuf_stream(
                                    onnx_file.filename, onnx_file.stream)

    return 'OK', 200

@app.route('/merge_model', methods=['POST'])
def merge_model():

    onnx_file1 = request.files['file0']
    onnx_file2 = request.files['file1']
    timestamp = time.time()
    global onnx_modifier
    onnx_modifier, stream ,merged_name = onnxModifier.merge(
        onnx_file1.filename, onnx_file1.stream, 
        onnx_file2.filename, onnx_file2.stream, 
        "", str(int(timestamp)) + "_")
    
    return send_file(stream,
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name=merged_name)
    
@app.route('/append_model', methods=['POST'])
def append_model():
    method = request.form.get('method')
    onnx_file1 = request.files['file']
    timestamp = time.time()
    global onnx_modifier
    if onnx_modifier:
        onnx_modifier, stream ,merged_name = onnx_modifier.append(
            onnx_file1.filename, onnx_file1.stream, 
            str(int(timestamp)) + "_", int(method))

    return send_file(stream,
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name=merged_name)


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    modify_info = request.get_json()

    global onnx_modifier
    onnx_modifier.reload()   # allow downloading for multiple times
    onnx_modifier.modify(modify_info)
    save_path = onnx_modifier.check_and_save_model()

    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='The hostname to listen on. \
                              Set this to "0.0.0.0" to have the server available externally as well')
    parser.add_argument('--port', type=int, default=5000,
                        help='The port of the webserver. Defaults to 5000.')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Enable or disable debug mode.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
