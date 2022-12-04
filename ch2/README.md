- **we need python3**
- if statements, vars ... python has no integer restrictions ... dynamicly typed vars 
- ''' quotes are for documentation
- **lists** are a primitive data type. Grouped collection of data. ['test', 'test 2']
  - lists have methods
   - append(var)
   - index(var) will check type and value.
   - negative index means end : -1
   - ranges [a:b] . one less than "b"
   - [a,b) b-nth is not included.
   - **in** keyword : x in b will check a list for a value.
    - b = [2,3,4,5]
    - 6 in b # false
   - **None** : null
   - a = [1,2,3,4]; b = a; b == [1,2,3,4]
   - [:] -- **get all elements**
   - **tuples** are lists that cant be copied/modified
   - Dictionaries : keys that are associated with a value
    - .keys()s will return a lis of keys
    - "in" keyword will test if a key exists in a dictionary
- Control structures
  - if-elif-else
  - for loops
  - while loops
  - with statements
  - try-except

### if-elif-else
```
if (a < b):
  print('asdfasdf')
elfi (a > b):
  print('asdf)
else:
  print('else')
```
### for loop
```
for i in range(6):
  print(i)
```

```
  x = ['how', 'now', 'brown', 'cow']
  for i in x:
    print(i)
```

```
  for i,v in enumerate(x):
    print(i, v);
```
### looping over dictionary:
  ```
    d = {
      'a': 1,
      'b': 2,
      'c': 3
    }
    for i in d:
      print i #prints key
  ```
  
  ```
    # print key and value
    for i in d:
      print(i, d[i])
  ```
  Construct a quickarray
  ```
    import random
    a = []
    for i in range(1000):
      a.append(random.random())
  ```
  while loops
  ```
    i = 0
    while(i<4):
      print(i)
      i += 1
  ```
Break
```
i = 0
while True:
  print(i)
  i += 1
  if (i==4):
    break;
```

###Continue
 continue means moves to nex itteration
 like `break` statement.
  ```
    i = 0
    while True:
      print(i)
      i += 1
      if (i == 4):
        break
  ```
  ```
    for i in range(4):
      print(i)
      continue
      print("xyz") #this never happens
  ```

## Working with files

### with
  with statement opens a file calles by name. 
  ```
    with open('somefile.txt') as f:
      s = f.read()
    s #output here....
  ```
## try-except blocks
  ```
  try:
    x = 1.0/0.0
  except:
    import db; db.set_trace()
  ```
  db.set_trace() . set_trace function will be called to enter into a debuggingng enviroment.
## functions
```
  def myFunction(a, b):
    return a*b

  myFunction(4,5)
```
pythone allows inner functions
## Modules
  Namespaces are a bag that we can access.we can import specific parts of our project.
  ```
    import time
    time.time()
  ```
  ```
    from time inmport ctime, localtime
    ctime()
  ```

## Reference
```
One-dimensional slices
The general syntax for a slice is array[start:stop:step]. Any or all of the values start, stop, and step may be left out (and if step is left out the colon in front of it may also be left out):

A[5:]
array([5, 6, 7, 8, 9])
A[:5]
array([0, 1, 2, 3, 4])
A[::2]
array([0, 2, 4, 6, 8])
A[1::2]
array([1, 3, 5, 7, 9])
A[1:8:2]
array([1, 3, 5, 7])

```
https://scipy-cookbook.readthedocs.io/items/Indexing.html