import json
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def downloadModel():
    modelNodeStates = request.get_json()
    print(modelNodeStates)
    return 'OK', 200

if __name__ == '__main__':
   app.run()