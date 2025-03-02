from django.shortcuts import render
from django.http import JsonResponse
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponseRedirect
import onnxruntime as ort
import numpy as np
import os

from .form import NameForm

# Get the absolute path to the directory containing this Python script.
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file.
model_path = os.path.join(base_dir, '../modelML/rdr_diabetes.onnx')

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


# def index(request):
#     return render(request, "model/index.html")

# Fix csrf khi gọi API
@csrf_exempt
def user(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
     
        pregnancies = data["pregnancies"]
        glucose = data["glucose"]
        bloodPressure = data["bloodPressure"]
        skinThickness = data["skinThickness"]
        insulin = data["insulin"]
        bmi = data["bmi"]
        dpf = data["dpf"]
        age = data["age"]


       
        input_data = np.array([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, dpf, age]])
        input_data = input_data.astype(np.float32)
     
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name


        pred_onx = sess.run([label_name], {input_name: input_data})[0]

       
        pred_onx = int(pred_onx[0])
       
        return JsonResponse({
                'status': status.HTTP_200_OK,
                'result': pred_onx,
        })
    
    return JsonResponse({
        'status': status.HTTP_400_BAD_REQUEST,
        'message': 'Invalid request method',
        })

def test(request):
    return JsonResponse({'status': status.HTTP_200_OK, 'message': 'Test'})

def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect("/thanks/")

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()

    return render(request, "model/index.html", {"form": form})