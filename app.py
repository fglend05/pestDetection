import sys, os
from pestDetection.pipeline.training_pipeline import TrainPipeline
from pestDetection.utils.main_utils import decodeImage, encodeImageIntoBase64, getPredictedLabels
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS, cross_origin
from pestDetection.constant.application import APP_HOST, APP_PORT
import cv2
import shutil
import time

app = Flask(__name__, static_url_path='/static')
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successful!!"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detection")
def detection():
    return render_template("detection.html")

@app.route("/e-manual")
def emanual():
    return render_template("e-manual.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/sample")
def sample():
    return render_template("sample.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_dir = f"static/results/{timestamp}"
        os.makedirs(result_dir, exist_ok=True)


        
        best_pt_path = "../best.pt"
        os.system(f"cd yolov5/ && python detect.py --weights {best_pt_path} --img 256 --conf 0.5 --source ../data/inputImage.jpg --save-txt")

        opencodedbase64 = encodeImageIntoBase64("yolov5/runs/detect/exp/inputImage.jpg")
        class_names = [
            "BitterGround_DowneyMildew",
            "BitterGround_FusariumWilt",
            "BitterGround_PowderyMildew",
            "Eggplant_PhomopsisFruitRot",
            "Eggplant_PhytophthoraBlight",
            "Eggplant_PowderyMildew",
            "Mustart_AlternariaLeafSpot",
            "Mustart_DownyMildew",
            "Mustart_WhiteRust",
            "Petchay_Blight",
            "w",
            "Petchay_PhytophthoraBlight",
            "Petchay_SoftRot",
            "StringBeans_BacterialBlight",
            "StringBeans_BeanRust",
            "StringBeans_Mosaic",
            "MelonFruitfly",
            "MelonWorm",
            "WhiteFly",
            "Aphids",
            "ColoradoPotatoBeetle",
            "FleaBeetles",
            "CutWorms",
            "DiamondbackMoth",
            "Beetles",
            "BeanFly",
        ]
        predicted_labels = getPredictedLabels("yolov5/runs/detect/exp/labels/inputImage.txt", class_names)

        result = {"image": opencodedbase64.decode('utf-8'), "predicted_labels": predicted_labels}

        # Create a ZIP archive of the prediction results
        archive_path = f"{result_dir}/exp.zip"
        shutil.make_archive(f"{result_dir}/exp", 'zip', "yolov5/runs/detect/exp")

        # Copy the results to static/history without zipping
        history_dir = f"static/history/{timestamp}"
        shutil.copytree("yolov5/runs/detect/exp", history_dir)

        # Clean up the runs directory
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

@app.route("/download")
def download():
    return render_template("download.html")

@app.route("/past_predictions")
def past_predictions():
    result_dir = "static/results/"
    predictions = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    return render_template("past_predictions.html", predictions=predictions)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame with YOLOv5
            # Save the frame as a temporary file
            cv2.imwrite("data/live_frame.jpg", frame)

            # Run YOLOv5 on the frame
            os.system(f"cd yolov5/ && python detect.py --weights best.pt --img 256 --conf 0.5 --source ../data/live_frame.jpg --save-txt")

            # Read the processed frame
            processed_frame = cv2.imread("yolov5/runs/detect/exp/live_frame.jpg")

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            # Yield the frame in a suitable format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Clean up the runs directory
            os.system("rm -rf yolov5/runs")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    clApp = ClientApp()
    print(f"Running on http://localhost:{APP_PORT}")
    app.run(host=APP_HOST, port=APP_PORT)
