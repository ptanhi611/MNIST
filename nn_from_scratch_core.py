class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)




class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0  




class Activation_sigmoid:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.array(1/(1+np.exp(-inputs)))
    
    def backward(self):
        self.dinputs=self.output*(1-self.output)




class Activation_Softmax:
    def forward(self, inputs):
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

    def backward(self, dvalues, y_true): 
        samples = len(dvalues)
        self.dinputs = dvalues.copy()

        if y_true.ndim == 1:
            self.dinputs[range(samples), y_true] -= 1
        else:
            self.dinputs -= y_true

        self.dinputs /= samples





class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        clipped_y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        if len(y_true.shape) == 1:
            correct_confidences = clipped_y_pred[range(len(y_pred)), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(clipped_y_pred * y_true, axis=1)
        return np.mean(-np.log(correct_confidences))





class Optimizer_SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases
