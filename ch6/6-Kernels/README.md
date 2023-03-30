# Support Vector Machines 127
https://www.youtube.com/watch?v=OKFMZQyDROI

C -  fudge factor affects the size of the margin found
y (gamma) - this parameter relates to how spread out the Guassina Kernel is around a particular training point.

Typically one uses a grid search over C and if using RBF kernel gamma to locate the best perfoming model.

We break up our data to higher dimensions to make it easier to seperate and find the margin
We take our our original data, we take original datas inner products, then we pass through kernel funciton and we get the same terms if we transform our data.


### Kernels
SVM uses inner products. We transpose a vector and apply matrix multiplication.

The kernel is this prodcut multiplication.

Linear kernel - vector transposed (row)

Other kernels are possible

#### Guassian Kernel (RBF)
Radial basis function Kernel. 

This kernel introduces a new parameter ()- it relates to how spread out the Gaussian Kernel is around a praticular training point.

One uses grid search over C and if using the RBF Kernel parameter, to locate the best perfoming model.

Support Vector Machines use training data mpped through a kernel function, to optimize the orientation and location of a hyperplane, that produces maximum margin between the hyper plane and the support vectors of the training data. The user needs to select the kernel function and associated parameters so that the model best fits the training data.