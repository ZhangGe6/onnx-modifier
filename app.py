from flask import Flask, render_template, request
import json
from onnx_modifier import onnxModifier
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/return_file', methods=['POST'])
def return_file():
    # https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
    onnx_file = request.files['file']
    
    global onnx_modifier
    onnx_modifier = onnxModifier.from_name_stream(onnx_file.filename, onnx_file.stream)
    
    return 'OK', 200


@app.route('/download', methods=['POST'])
def modify_and_download_model():
    node_states = json.loads(request.get_json())
    
    print(node_states)
    onnx_modifier.remove_node_by_node_states(node_states)
    onnx_modifier.check_and_save_model()
    
    
    return 'OK', 200

if __name__ == '__main__':
   app.run()