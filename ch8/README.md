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
