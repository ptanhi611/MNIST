import pickle
from nn_from_scratch_core.layers import Layer

def load_model(hidden_layers,input_size,output_size,filename):
    layer_size=[input_size]+hidden_layers+[output_size]
    Layers=[Layer(layer_size[i], layer_size[i+1]) for i in range(len(layer_size)-1)]

    with open(filename,"rb") as f:
        params=pickle.load(f)
    
    for i,layer in enumerate(Layers):
        layer.weights=params["weights"][i]
        layer.biases=params["biases"][i]
    
    
    
    return Layers
