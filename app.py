from predict import predict

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "hello, world"


@app.route('/predict', methods=['POST'])
@cross_origin()
def runPredict():
    req_data = request.get_json(force=True)
    result = predict(req_data)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)