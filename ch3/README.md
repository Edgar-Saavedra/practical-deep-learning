# Using Numpy

[Numpy Documentation](https://numpy.org/doc/stable/user/quickstart.html)

Lists vs Arrays

**array** a fixed-sized block of contiguous memory. A single block of RAMwith no gaps. Quick to index.
**multidimensional** is also fixed size. Tow numbers are needed to determine the locaions of a piece (row and column)
**stacking multidimensional** we have a stack, with rows  + columns
**list** not contiguous memory but scattered throughout RAM with pointers linking one element to next like in a chain.
**vector** one dimensional array

## NUMPY
Allows to quickly operat on array data. All about arrays.

### Methods & Keys:
- **array()** takes arguement to turn into an array.
- **numpy.size** number of elements in array
- **.shape** a tuple of number of arrays in each dimension
- **.dtype** the data type of the array values [More on dtypes](https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes)

### Defining with 0's and 1's
**numpy.zeros(tuple(zise, row, colum ...))** numpy workhorse function tha returns new arrays with every element set to 0.
**numpy.ones(tuple(zise, row, colum ...))** numpy workhorse function tha returns new arrays with every element set to 1.
**numpy.astype()** returns a copy of the array casint each element to the given data type
**numpy.copy()** if you wan to actully create a new copy of a Numpy array use the copy method
**numpy.arange()** analogue to `range()` returns a vector of given size [docs](https://numpy.org/doc/stable/reference/generated/numpy.arange.html?highlight=arange#numpy.arange)
**slicing** is done is a similar fashion using array notaiont [x:] [:x] [x:y:z]
**numpy.reshape()** change the vector in a matrix of x:y:... `b = np.arange(20).reshape((4,5))`
**numpy.dot()** to multiply two vectors, a vector and a matrix or two matrices, following linear algebra rules

### Accessing elements
We index numpy arras in the same way we index lists with square brakeds

### broadcasting
To decide how to apply an operation to arrays. Aka you can apply an arthmatic operation to all ements

### Saving data
**np.loadtxt()** To load a text file, like csv
**np.save(PATH, array)** To save data to a `npy` file
**np.load()** To load an`npy` file
**np.savez(PATH, ...)** to save multiple arrays to disk `np.savez("arrays", a=a, b=b)`. You get to set the keys you want to save to
`q = np.load("arrays.npz")` `q['a']`

### Random numbers
**random.normal** Random numbers drawn from a bell-shaped curve
**random.seed** set the seed of the generator so we can produce the same sequence

### Numpy and images
- use pillow module (PIL) to work with images.
- WE convert images to numpy arrays. We will use PIL to read and write image files
- Images are 2-d arrays of numbers, with color however it is 3-4 numbers for each pixel (RGB)
- **PIL.Image.show()**
- **PIL.Image.save()**
- **PIL.Image.open()**
- **PIL.Image.convert("L")** Convert to grayscale
- [pillow website](https://pillow.readthedocs.io/en/stable/)