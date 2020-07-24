from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import json
import pickle
from file_operations import file_methods
from application_logging import logger

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
app = application
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict_json", methods=['POST'])
@cross_origin()
def predictJsonRouteClient():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # predicting for dataset present in database
            path, json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!" + str(path) + ' and few of the predictions are ' + str(json.loads(json_predictions)))
        else:
           print('Nothing Matched')

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        path = request.form['Default_File_Predict']

        pred_val = pred_validation(path)  # object initialization

        pred_val.prediction_validation()  # calling the prediction_validation function

        pred = prediction(path)  # object initialization

        # predicting for dataset present in database
        path, json_predictions = pred.predictionFromModel()

        return render_template('results.html',prediction='Prediction has been saved at {} and few of the predictions are '.format(path) +' ' + str(json.loads(json_predictions)))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/predict_new", methods=['POST'])
@cross_origin()
def predictNewRouteClient():
    try:
        age_old = int(request.form['age'])
        age = round(age_old/365)
        is_gender = request.form['gender']
        if (is_gender == 'Male'):
            gender = 0
        else:
            gender = 1
        height=int(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        is_cholesterol = request.form['cholesterol']
        if (is_cholesterol == 'Low'):
            cholesterol = 1
        elif(is_cholesterol == 'Medium'):
            cholesterol = 2
        else:
            cholesterol = 3
        is_gluc = request.form['gluc']
        if (is_gluc == 'Low'):
            gluc = 1
        elif (is_gluc == 'Medium'):
            gluc = 2
        else:
            gluc = 3
        is_smoke = request.form['smoke']
        if (is_smoke == 'Yes'):
            smoke = 1
        else:
            smoke = 0
        is_alco = request.form['alco']
        if (is_alco == 'Yes'):
            alco = 1
        else:
            alco = 0
        is_active = request.form['active']
        if (is_active == 'Yes'):
            active = 1
        else:
            active = 0
        BMI= weight / ((height/100)**2)

        filename = "models/KMeans/KMeans.sav"
        loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
        # predictions using the loaded model file
        clusters=loaded_model.predict([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, BMI]])
        file_object = open("Prediction_Logs/Prediction_Log_single.txt", 'a+')
        log_writer = logger.App_Logger()
        file_loader = file_methods.File_Operation(file_object, log_writer)

        model_name = file_loader.find_correct_model_file(clusters[0])
        model = file_loader.load_model(model_name)
        result = model.predict([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, BMI]])
        if (result[0] == 0):
            output = 'Low(0)'
        else:
            output = 'High(1)'
        log_writer.log(file_object, 'End of Prediction')
        file_object.close()

        return render_template('results.html',prediction='Your chance of having Cardio Vascular Disease is {}'.format(output))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function

            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #host = '0.0.0.0'
    #port = 5000
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()
    app.run(debug=True)
