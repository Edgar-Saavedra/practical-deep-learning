# NEURAL NETWORKS

We will present the components of a fully connected feed-forward NN. Discuss the structure and parts of NNs. Explore trainig example to classify irises.

aka Artificial Neural Networks
aka Multi-layer perceptrons (MLPs) `MLPClassifier` class

## Anotomy of a NN

- A neural network is a graph
  - A series of Nodes (drawn as circles) connected by Edges (short line segments)
- Neural Networks are function approximations
  - Use graph strucuture to represent a series of computed steps.
  - The steps map an input feature vector to and output value, typically interpreted as a probility.
- Neural Networks are built in layers.
  - They are from left to right mapping input feature vector to the outputs. 
  - Passing values along the edges to the nodes.
- Nodes
  - Also known as neurons
  - Nodes Calculate the new values based on their inputs
  - Pass the the next layer of nodes until the output nodes are reached.
- Fully Connected Feed-Forward Neural Network
  - Every node of a layer has its output sent to every node of the next layer.
  - Feedforward - from left to right, without being sent back. No feedback. No looping.

### Neuron

Accepts inputs from the left and has a single output on the right. Could accept hundreds of inputs.
This echos how a neuron in brain work: many inputs mapped to a single output.

* A neural node accepts multiple inputs x0,x1 multiplies each by a weight w0,w1 sums these products along the bias term (b) passes this sum to the activiation function to produce a single scalar output value (a)
* Feed it a feature vector and out comes a classification.

Example:
  - Dendrite : accept input from many other neurons
  - Axon: is the output.

#### Activation Function (h)

Calculates the output of the node, a single number.
Inputs are represented as squares.

- Inpute Scalar (x): Each input is a single number.
- Line Segments (w): the input move to the node along the line segments. 
  - Line segments represent weights, strength of the connection
  - Inputs are multiplied by the weights, summed and given to activation function.
- Output (a)

The inputs multiplied by the weights are added together and given to the activation function to produce an output value.

- Bias term (b)
  - an offset used to adjust the activation function.

#### Activation Functions

We need the activation funciton to be non-linear so the model can learn complex functions. This means that there one weight and one bias value per node.

- Linear Function
  - Has an output that is directly proportional to the input.
  - Example g(x) = 3x + 2
  - it is usually a straight line
- Non Linear Function
  - Example: g(x) = x^2 + 2
  -  Transcendental functions are also non-linear : log(x), e^x
  - Trigonometric functions : sine, cosine, tangent
  - Their graphs are not straight lines

Traditional NNs use sigmoid or hyperbolic tangets for activation functions

sigmoid
`o(x) = 1/(1+e^-x)`

Hyperbolic
`tanh(x) = (e^x - e^-1)/ (e^x + e^-x) = (e^2x-1)/(e^2x+1)`

Both of these functions have roughly the same S shape.

More Recently the sigmoid and hyperbolic tanget have been replaced with the rectified linear unit (ReLU)

ReLu - is not a straight line. Used in backpropagation training of NNs

### ReLu
Rectified Linear Unit, a non linear activation funciton. It is called rectified because it removes the negative values and replaces them with 0. It is computationally simple, faster to calculate than either sigmoid or hyperbolic tangent.

### Architecture of a NN 
Standard Networks Are built in layers.

FeedForward Network 
  - has an input layer (feature vector)
  - one or more layers
    - hidden layers are mde of nodes.
  - an output layer (prediction)
  - Nodes
    - accept an input the output of the nodes of layer `i-1` and pass their output to the inputs of node `i+1`
    - Connections between layers are usually fully connected
    - The number of nodes in each hidden layer defin the architecture of the network.
    - i has been proven that a single hidden layer with enough nodes can learn any function mapping.
    - The model acts as a complex function mapping inputs to output labels and probabilities.
  - As the number of nodes (and layers) grows, so too does the number of parameters to learn (weights and biases) and therefore also the amount of training data - Cures of dimensionality

### Selecting The Right Achitecture
- if your input has a definite spatial relationship (image), you might want to use a CNN (convolutional neural network)
- use no more than 3 hidden layers. 1 sufficiently large hidden layer is all that is needed. If it learns with 1 then maybe see about adding a second hidden layer.
- number of nodes in the first hidden layer should match or exceed the number of input vector features
- Except for first hidden layer, the number of nodes per hidden layer should be the same or some value between the number of nodes in the previous layer and following layer.

1. NNs best apply to situations where your input does not have spatial relationship - not an image.
2. Input decision is small, not a lot of data - not good for CNNs.

## Output layers
If the NN is modeling a continous value - regression, then the output is a node that doesn't use an activation function.

If NN is for `single classification`, we wnat them to output a decision value. If we have 2 classes (class 0, class 1), we make the activation function of the final node a sigmoid. It will output a value in that range. If the activation function value is less than 0.5 call that class 0.

If NN is for `multiple classification`, instead of single node output, we'll have `N` output nodes. One for each class, each one using the `identity function` for `h` then we apply a `softmax operation` the these `N` outputs and select the output with thte largest `softmax` value  
  
## Softmax

Example: Suppose we have a datase with 4 classes (labeled 0, 1, 2, 3) `N=4` our network will have 4 output nodes each using the `identity function` for `h`. We select the largest value in this output vector as the class label for the given input. 

Softmax ensures that the elements of this vector sum to 1. These are the probability of belonging to each of the four classes. That is why we take only the largest value.


```
softmax 179
p_i = e^a_i / E_je^a_j
```

## Representing Weights and Biases

We can view weights and biases in terms of matrices and vectors

Weights and Biases of a NN can be stored in NumPy Arrays and we need only simple matrix operations `np.dot` and addition to work with a fully conencted NN.

We need a `weight matrix` and a `bias vector` between each layer.