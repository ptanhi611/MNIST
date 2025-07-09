# model/predict_single.py

import numpy as np
from model_loader import load_model
from model import Activation_ReLU, Activation_Softmax

def predict_single_digit(image_vector, model_path, hidden_layers, output_size):
    input_size = image_vector.shape[1]
    Layers = load_model(hidden_layers, input_size, output_size, model_path)

    activation1 = Activation_ReLU()
    activation2 = Activation_Softmax()

    current_input = image_vector
    for layer in Layers[:-1]:
        layer.forward(current_input)
        activation1.forward(layer.output)
        current_input = activation1.output

    Layers[-1].forward(current_input)
    activation2.forward(Layers[-1].output)

    predictions = activation2.output
    predicted_label = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    return predicted_label, confidence, predictions
