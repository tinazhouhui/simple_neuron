import numpy as np

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

expected_output = np.array([
    [0],
    [1],
    [1],
    [0],
])

epochs = 100000
learning_step = 0.1

input_layer_neurons = 2
hidden_layer_neurons = 2
output_layer_neurons = 1

hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
# print("Hidden weights:")
# print(hidden_weights)
output_weights = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
# print("Output weights:")
# print(output_weights)

hidden_biases = np.random.uniform(size=(1, hidden_layer_neurons))
# print("Hidden biases:")
# print(hidden_biases)
output_biases = np.random.uniform(size=(1, output_layer_neurons))
# print("Output biases:")
# print(output_biases)


def sigmoid(layer_activation):
    return 1 / (1 + np.exp(-layer_activation))


def sigmoid_derivative(x):
    return x * (1 - x)


for i in range(epochs):
    # forward propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)# + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)# + output_biases
    predicted_output = sigmoid(output_layer_activation)

    # back propagation
    error = expected_output - predicted_output
    predicted_output_derivative = sigmoid_derivative(predicted_output) * error

    hidden_layer_error = predicted_output_derivative.dot(output_weights.T)
    hidden_layer_derivative = sigmoid_derivative(hidden_layer_output) * hidden_layer_error

    output_weights += hidden_layer_output.T.dot(predicted_output_derivative) * learning_step
    #output_weights[0] += np.average(predicted_output_derivative) * learning_step
    #output_weights[1] += np.average(predicted_output_derivative) * learning_step

    hidden_weights += inputs.T.dot(hidden_layer_derivative) * learning_step

print(predicted_output)

