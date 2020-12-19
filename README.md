# Simple neural network
The goal is to create a neural network to learn XOR based on two inputs 
that can take value of 1 or 0. 

For this simplified method, it is possible to define _all_ possible inputs.

```python
input = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
```
For each of the input, it is important to specify expected output. 
```python
expected_output = [
    [0],
    [1],
    [1],
    [0],
]
```
Activation function used was Sigmoid function 

### Requirements
Numpy

## Run
```python
python3 neuron.py
```
