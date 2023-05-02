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

We want a model that makes few mistakes. We need a set of parameters that lead to a small value for the `loss function`

We need to find a `minimum of the loss function`

We need an algorithm that finds the minimum of a function, by picking a starting point and use the gradient to move to a lower point.

The `gradient` tells us in what direction to move.

`step size` tells us how big of a jump we make from on `x` position to the next. It is a parameter we have to choose, it is also called the `learning rate`.  `Learning rate` is often fluid and gets smaller as we move - assumption is that we get closer to the minimum.

`local minimum` a low value that is not the `minimum`.

The `gradient` tells us how small change in `x` changes `y`. If `x` is one of the parameters of our network and `y` is the error given by the `loss function` the the gradient tells us how much a change in that paramter affects the overall error. Once we know the error we are in a position to modify the paremeter by and amount based on the gradient - which will move us towards a minimum. When the error over the training set ist at a minimum we claim the network is trained.

Every `weight` and `bias` in our network is a parameter, and the `loss function` value depends upon all of them. No matter the dimensions, if we know the gradient of each parameter we can still apply our algorithm in an attemp to locat a set of parameter minimizing the `loss function`.

## Updating weights

Assuming we already have `gradient` values. Assume we have a set of numbers that tell us how to get `gradient`, a change in any weight or bias changes the loss function. We can then apply `gradient descent` - we adjust `weights` and `biase` by some fraction to move us toward a `minimum` of the `loss function`

We update each `weight` and `bias` by this rule:

`w <- w <- n∆w`

w - one of the weights (or bias)
n - eta , the learning rate, ∆w gradient value

### Algorithm training gradient descent

```
1. Pick some intelligent starting values from the weights and biases
2. Run the trainig set through the network using its current weights and biases and calcualte the average loss.
3. Use this loss to get the gradient for each weight and bias.
4. Update the weight or bias value by the step size times the gradient value.
5. Repeate from step 2 until loss is low enough
```

A succesfull NN relies  on choosing a good initial value.
 
Step 2, is the `forward-pass` through the network 
step 3 is a black box. 
step 4 moves the parameter form its current value to one that will reduce the overall loss.
There are other terms like `momentum` that preserv some fraction of the previous weight change for the next iteration, so that parameters don't change wildly.


