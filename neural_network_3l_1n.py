import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# This is derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
def der_sigmoid(x):
   return sigmoid(x)*(1-sigmoid(x))

data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_target = [0,1,1,1]

data_shape = data[0].shape
a0 = np.random.normal(size = data_shape[0])
b0 = np.random.normal()

a1 = np.random.normal()
b1 = np.random.normal()

a2 = np.random.normal()
b2 = np.random.normal()
learning_rate = 0.01
epochs = 5000

for epoch in range(epochs):
    
    for x,y in zip(data,y_target):
        
        ## This is 3 layer network with 1 neuron in each layer a very simple neural network

        # Layer 0: Input to first hidden layer
        # h0 = (x1 * a1) + (x2 * a2) + b0 ---- here a are the weights of the neuron
        hidden_layer_0 = np.dot(x,a0) + b0
        hidden_layer_output_0 = sigmoid(hidden_layer_0)

        # Layer 1: Hidden_0 to Hidden_1
        # h1 = (hidden_layer_output_0 * a1) + b1
        hidden_layer_1 = np.dot(hidden_layer_output_0,a1) + b1
        hidden_layer_output_1 = sigmoid(hidden_layer_1)

        # Layer 2: Hidden_1 to Output
        # h2 = (hidden_1 * a2) + b2
        hidden_layer_2 = np.dot(hidden_layer_output_1,a2) + b2
        hidden_layer_output_2 = sigmoid(hidden_layer_2)

        y_pred = hidden_layer_output_2

        loss = (y_pred - y)**2

        # Output layer gradients:
        # ∂L/∂a2 = ∂L/∂y_pred*∂y_pred/∂(hidden_layer_2)*∂(hidden_layer_2)/∂a2
        # ∂L/∂a2 = 2(ŷ - y) * σ'(h2) * h1
        dl_da2 = 2*(y_pred - y)*der_sigmoid(hidden_layer_2)*hidden_layer_output_1
        dl_db2 = 2*(y_pred - y)*der_sigmoid(hidden_layer_2)

        # ∂L/∂a1 = ∂L/∂y_pred*∂y_pred/∂(hidden_layer_2)*∂(hidden_layer_2)/∂(hidden_layer_output_1)*∂(hidden_layer_output_1)/∂(hidden_layer_1)*∂(hidden_layer_output_1)/∂a1
        # ∂L/∂a1 = 2(ŷ - y) * σ'(h2) * a2 * σ'(h1) * h0
        dl_da1 = 2*(y_pred - y)*der_sigmoid(hidden_layer_2)*a2*der_sigmoid(hidden_layer_1)*hidden_layer_output_0
        dl_db1 = 2*(y_pred - y)*der_sigmoid(hidden_layer_2)*a2*der_sigmoid(hidden_layer_1)

        dl_da0 = 2*(y_pred - y)*der_sigmoid(hidden_layer_2)*a2*der_sigmoid(hidden_layer_1)*a1*der_sigmoid(hidden_layer_0)*x
        dl_db0 = 2*(y_pred - y)*der_sigmoid(hidden_layer_2)*a2*der_sigmoid(hidden_layer_1)*a1*der_sigmoid(hidden_layer_0)

        # Gradient descent update 
        a2 = a2 - learning_rate*dl_da2
        b2 = b2 - learning_rate*dl_db2

        a1 = a1 - learning_rate*dl_da1
        b1 = b1 - learning_rate*dl_db1

        a0 = a0 - learning_rate*dl_da0
        b0 = b0 - learning_rate*dl_db0

    if epoch % 10 == 0:
        print("Epoch %d loss: %.3f" % (epoch, loss))

