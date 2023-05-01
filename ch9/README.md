# Theory behind neural networks

## The Overall process

1. First step in training a neural network is selecting `intelligent initial values` for the `weights` and `biases`.
2. Then use `gradien descent` to modify these `weights` and `biases` so that we reduce the error over the training set.
3. Use `average value` of the `loss function` to `measure the error` - tells us how wrong the network is.
4. We know the network is right or wrong because we have the expected output for each input sample in the training set.

## Gradient Descent, Backpropagation, loss functions, weight initialization, regularization

- Gradient Descent - pick gradients for weights and biases
- Loss function - the measure of error
- Backpropagation - Gives the gradients starting at the output moving back through network.
- Regularlization - to help with overfitting and generalization.

An algorithm that requires gradients. Think of gradients as measures of steepness. The large the gradient the steepeer the function is at that point.

To use `gradient descent` to search for the smallest value of the loss function, we need to be able to find gradients. For that we use `back propagation`. 

`back propagation` is the fundamental algortihm of NNs - allow them to learn succesfully. It gives us the gradients we need to starting at the output of the network and moving back through the network toward the input. Along the way it calculates the `gradient value` for each `weight` and `bias`.

With `gradient values` we can use `gradient descent` to update the weights `weights` and `biases` so that the next time we pass the training samples  throught the network the `average of the loss function` will be less than it was before. Our network will be less wrong.

Learnng general features of the dataset requires `regularization` - helps with overfitting.

## Gradient Descent

Standard way to train a NN.

Descent - go down from somewhere higher up
Gradient - How quickly something changes with respect to how fast something else changes. Example: miles per hour etc

### Consider equation of a line
`y = mx + b`

m - slope (how quickly the lines y position changes with each change in x)
b - y-axis intercept

### Calculatin slope

`m = y0 - y1/x0 - x1`

y's per x, how steep or shallow the line is : its gradient

A change in a variable is noted as ∆ (delta)

`m = ∆y/∆x`

Most functions have a slope at each point. However the slope changes from point to point. `Tangent lines` are the lones that touch the function at one particullar point.

How the slope changes over the funciton is itself a funciton : `derivative`. Given a function and x value the derivative tells us the slope of te fucntion at a point x.

We want to find the `minimum of the function` - the x that gives us the smalles y. We want to move in the `opposite` direction to the gradient as that will move us in the directions fo the `minimum`

Derivative:
`dy/dx`


## Finding Minimums