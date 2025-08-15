import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# This is derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
def der_sigmoid(x):
   return sigmoid(x)*(1-sigmoid(x))

data = np.array([[0,0,0,0],[0,1,0,0],[1,0,0,0],[1,1,0,0]])
y_target = [0,1,1,1]

data_shape = data[0].shape
a00 = np.random.normal(size = data_shape[0])
b00 = np.random.normal()

a01 = np.random.normal(size = data_shape[0])
b01 = np.random.normal()

a02 = np.random.normal(size = data_shape[0])
b02 = np.random.normal()


a10 = np.random.normal(size = 3)
b10 = np.random.normal()

a11 = np.random.normal(size = 3)
b11 = np.random.normal()


a20 = np.random.normal(size = 2)
b20 = np.random.normal()


learning_rate = 0.001

epochs = 500

for epoch in range(epochs):
    
    for x,y in zip(data,y_target):

        # This is 3 layer network with 3,2,1 neurons in layers respectively

        # Layer 0: Input to first hidden layer
        # h_00 first 0 is the layer number and 2nd zero is neuron number --- refer neural_network_3l_1n.py for previous... 
        # ...comments as that network was more simpler

        h_00 = np.dot(x,a00) + b00
        z_00 = sigmoid(h_00)

        h_01 = np.dot(x,a01) + b01
        z_01 = sigmoid(h_01)

        h_02 = np.dot(x,a02) + b02
        z_02 = sigmoid(h_02)

        # Layer 1: Hidden_0 to Hidden_1
        h_10 = np.dot([z_00,z_01,z_02],a10) + b10
        z_10 = sigmoid(h_10)

        h_11 = np.dot([z_00,z_01,z_02],a11) + b11
        z_11 = sigmoid(h_11)

        # Layer 2: Hidden_1 to Output
        h_20 = np.dot([z_10,z_11],a20) + b20
        z_20 = sigmoid(h_20)

        y_pred = z_20

        loss = (y_pred - y)**2

        # Output layer gradients:
        # ∂L/∂a2 = ∂L/∂y_pred*∂y_pred/∂(hidden_layer_2)*∂(hidden_layer_2)/∂a2
        # ∂L/∂a2 = 2(ŷ - y) * σ'(h2) * h1

        dl_dypred = 2*(y_pred - y)
        dypred_dh20 = der_sigmoid(h_20)
        dh20_da200 = z_10
        dh20_db200 = 1
        dh20_da201 = z_11
        dh20_db201 = 1


        #dl_da20 = np.array([dl_dypred*dypred_dh20*dh20_da200, dl_dypred*dypred_dh20*dh20_da201])
        dl_da20 = (dl_dypred*dypred_dh20)*np.array([z_10,z_11])
        dl_db20 = dl_dypred*dypred_dh20

        a201 = a20[1]
        dh20_dz11 = a201
        dz11_dh_11 = der_sigmoid(h_11) 
        dh_11_da110 = z_00
        dh_11_da111 = z_01
        dh_11_da112 = z_02

        #dl_da11 = np.array([dl_dypred*dypred_dh20*dh20_dz11*dz11_dh_11*dh_11_da110,dl_dypred*dypred_dh20*dh20_dz11*dz11_dh_11*dh_11_da111,dl_dypred*dypred_dh20*dh20_dz11*dz11_dh_11*dh_11_da112])
        dl_da11 = (dl_dypred*dypred_dh20*dh20_dz11*dz11_dh_11)*np.array([z_00,z_01,z_02])
        dl_db11 = dl_dypred*dypred_dh20*dh20_dz11*dz11_dh_11


        a200 = a20[0]
        dh20_dz10 = a200
        dz10_dh_10 = der_sigmoid(h_10) 
        dh_10_da100 = z_00
        dh_10_da101 = z_01
        dh_10_da102 = z_02

        dl_da10 = (dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10)*np.array([z_00,z_01,z_02])
        dl_db10 = dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10

        a102 = a10[2]
        dh_10_dz02 = a102
        dz02_dh_02 = der_sigmoid(h_02)
        # dh_02_da020 = x[0]
        # dh_02_da021 = x[1]
        # dh_02_da022 = x[2]
        # dh_02_da023 = x[3]


        dl_da02 = (dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz02*dz02_dh_02)*x
        dl_db02 = dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz02*dz02_dh_02

        a101 = a10[1]
        dh_10_dz01 = a101
        dz01_dh_01 = der_sigmoid(h_01)
        # dh_01_da010 = x[0]
        # dh_01_da011 = x[1]
        # dh_01_da012 = x[2]
        # dh_01_da013 = x[3]


        dl_da01 = (dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz01*dz01_dh_01)*x
        dl_db01 = dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz01*dz01_dh_01


        a100 = a10[0]
        dh_10_dz00 = a100
        dz00_dh_00 = der_sigmoid(h_00)
        # dh_00_da000 = x[0]
        # dh_00_da001 = x[1]
        # dh_00_da002 = x[2]
        # dh_00_da003 = x[3]


        #dl_da00 = np.array([dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz00*dz00_dh_00*dh_00_da000,dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz00*dz00_dh_00*dh_00_da001,dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz00*dz00_dh_00*dh_00_da002,dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz00*dz00_dh_00*dh_00_da003])
        # simplified
        dl_da00 = (dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz00*dz00_dh_00)*x
        dl_db00 = dl_dypred*dypred_dh20*dh20_dz10*dz10_dh_10*dh_10_dz00*dz00_dh_00

        # Gradient descent update 
        a20 = a20 - learning_rate*dl_da20
        b20 = b20 - learning_rate*dl_db20

        a11 = a11 - learning_rate*dl_da11
        b11 = b11 - learning_rate*dl_db11

        a10 = a10 - learning_rate*dl_da10
        b10 = b10 - learning_rate*dl_db10

        a00 = a00 - learning_rate*dl_da00
        b00 = b00 - learning_rate*dl_db00

        a01 = a01 - learning_rate*dl_da01
        b01 = b01 - learning_rate*dl_db01


        a02 = a02 - learning_rate*dl_da02
        b02 = b02 - learning_rate*dl_db02



    if epoch % 50 == 0:
        print("Epoch %d loss: %.3f" % (epoch, loss))

