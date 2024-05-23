import sys,os
from pestDetection.pipeline.training_pipeline import TrainPipeline
from pestDetection.utils.main_utils import decodeImage, encodeImageIntoBase64, getPredictedLabels
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from pestDetection.constant.application import APP_HOST, APP_PORT
import cv2
import json

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.prediction_history = []

@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/history", methods=['GET'])
@cross_origin()
def getHistory():
    history_file_path = "history/prediction_history.json"
    try:
        with open(history_file_path, "r") as file:
            history = json.load(file)
    except FileNotFoundError:
        history = []

    return jsonify(history)

def saveHistory(history):
    history_file_path = "history/prediction_history.json"
    with open(history_file_path, "w") as file:
        json.dump(history, file)

@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        os.system("cd yolov5/ && python detect.py --weights best.pt --img 256 --conf 0.5 --source ../data/inputImage.jpg --save-txt")

        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        class_names = [
            "dog", "person", "cat", "tv", "car", "meatballs", "marinara sauce", "tomato soup", "chicken noodle soup",
            "french onion soup", "chicken breast", "ribs", "pulled pork", "hamburger", "cavity", "Mustard_AlteriaLeafSpot",
            "Cucumber_AngularLeafSpot", "StringBeans_BacterialBlight", "StringBeans_BeanRust", "Cabbage_Blight",
            "Cabbage_AlteriaLeafSpot", "Cabbage_BlackRotXantomonas", "BitterGround_DowneyMildew", "StringBeans_Mosaic",
            "Eggplant_PhytophthoraBlight", "Cucumber_PhytophthoraBlight", "BitterGround_PowderyMildew", "Cucumber_PowderyMildew",
            "Eggplant_PowderyMildew", "Mustard_WhiteRust"
        ]
        predicted_labels = getPredictedLabels("yolov5/runs/detect/exp/labels/inputImage.txt", class_names)

        result = {"image": opencodedbase64.decode('utf-8'), "predicted_labels": predicted_labels}
        clApp.prediction_history.append(result)
        saveHistory(clApp.prediction_history)
        os.system("rm -rf yolov5/runs")

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)



@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 256 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    
if __name__ == "__main__":
    clApp = ClientApp()
    print(f"Running on http://localhost:{APP_PORT}")
    app.run(host=APP_HOST, port=APP_PORT)
