#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      s-bhoener
#
# Created:     14/02/2023
# Copyright:   (c) s-bhoener 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import random
import math
# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Define the matrix multiplication function
def matmul(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            sum = 0
            for k in range(len(b)):
                sum += a[i][k] * b[k][j]
            row.append(sum)
        result.append(row)
    return result

# Define the neural network architecture
def create_network(layer_sizes):
    network = []
    for i in range(len(layer_sizes) - 1):
        input_size = layer_sizes[i]
        output_size = layer_sizes[i+1]
        weights = [[random.random() for _ in range(output_size)] for _ in range(input_size)]
        layer = {"weights": weights, "activation": sigmoid}
        network.append(layer)
    return network

# Perform the forward pass of the neural network
def forward_pass(network, x):
    layer_input = x

    for layer in network:
        weights = layer["weights"]
        activation = layer["activation"]
        layer_output = matmul(layer_input, weights)
        layer_output = [[activation(val) for val in row] for row in layer_output]
        layer["output"] = layer_output
        layer_input = layer_output
    return layer_output

# Define the input to the neural network
x = [[random.random()]]

# Specify the number of neurons in each layer
layer_sizes = [1, 3, 4]

# Create the neural network
network = create_network(layer_sizes)

for y in range(1000):
    x = [[random.random()]]
    # Perform the forward pass of the neural network
    output = forward_pass(network, x)

    print("Input:", x)
    print("Output:", output)
