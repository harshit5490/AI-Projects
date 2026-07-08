import numpy as np
# print(np.__version__)

# number = np.array([10,20,30,40])
# print(type(number))
# print(number.shape)
# print(number.ndim)
# print(number.size)
# print(number.dtype)
# print(number.itemsize)
# print(number.nbytes)

arr = np.zeros(5)
# print(arr)
arr = np.zeros(5,dtype=int)
# print(arr)
arr = np.ones(5,dtype=int)
# print(arr)
arr = np.arange(5)
# print(arr)
arr = np.arange(2,10,2)
# print(arr)

arr = np.linspace(0,10,5)
# print(arr)

numbers = np.arange(1,13)

# print("Original")
# print(numbers)

# print("\nShape")
# print(numbers.shape)

# print("\n3 x 4")
# print(numbers.reshape(3,4))

# print("\n4 x 3")
# print(numbers.reshape(4,3))

# print("\n6 x 2")
# print(numbers.reshape(6,2))

# print("\nAuto Rows")
# print(numbers.reshape(-1,2))

# print("\nAuto Columns")
# print(numbers.reshape(3,-1))


# a = np.array([10,20,30,40,50])

# print("Original")
# print(a)

# print("\nIndexing")
# print(a[0])
# print(a[-1])

# print("\nSlicing")
# print(a[1:4])
# print(a[:3])
# print(a[2:])
# print(a[::2])

# print("\nMatrix")

# matrix = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# print(matrix)

# print("\nElement")
# print(matrix[1,2])

# print("\nRow")
# print(matrix[1])

# print("\nColumn")
# print(matrix[:,1])

# print("\nViews")

# b = a[1:4]

# b[0] = 999

# print(a)
# print(b)

# print("\nCopies")

# a = np.array([10,20,30,40,50])

# c = a[1:4].copy()

# c[0] = 777

# print(a)
# print(c)

salary = np.array([30000,40000,50000,60000])

new_salary = salary + 5000

# print(new_salary)


# a = np.array([10,20,30])
# b = np.array([1,2,3])

# print("Addition")
# print(a+b)

# print("\nSubtraction")
# print(a-b)

# print("\nMultiplication")
# print(a*b)

# print("\nDivision")
# print(a/b)

# print("\nPower")
# print(a**2)

# print("\nSquare Root")
# print(np.sqrt(a))

# print("\nScalar Broadcasting")
# print(a+100)

# print("\n2D Broadcasting")

# matrix = np.array([
#     [1,2,3],
#     [4,5,6]
# ])

# print(matrix+10)

# print("\nVector Broadcasting")

# vector = np.array([100,200,300])

# print(matrix+vector)

salaries = np.array([30000, 35000, 45000, 60000, 75000, 50000])
# print(salaries > 50000)
# print(salaries[salaries > 50000])
# print(salaries[(salaries > 40000) & (salaries < 70000)])
# print("Sum:", np.sum(salaries))
# print("Mean:", np.mean(salaries))
# print("Max:", np.max(salaries))
# print("Min:", np.min(salaries))
# print("Median:", np.median(salaries))
# print("Standard Deviation:", np.std(salaries))

a = np.random.rand(5)
# print(a)
a = np.random.randint(1,10,5)

# print(a)
matrix = np.random.randint(1,100,(3,4))

# print(matrix)

np.random.seed(42)

# print(np.random.randint(1,100,5)) this will always give the same random number because of seed

students = np.array([
    "A",
    "B",
    "C",
    "D",
    "E"
])
np.random.shuffle(students)

# print(students)

selected = np.random.choice(students,3)
# print(selected)

a = np.random.randn(5)
# print(a)

A = np.array([
    [1,2],
    [3,4]
])

B = np.array([
    [5,6],
    [7,8]
])

# print(A*B)

# print(A @ B) or print(np.matmul(A,B)) matrix multiplication

# print(A.T) Transpose

I = np.eye(3)
# print(I) Identity Matrix
# np.dot(A,B) Dot Product

employees = np.array([
    [80,90,85],
    [70,75,80],
    [95,88,92],
    [60,65,70]
])
weights = np.array([0.5,0.2,0.3])

print(employees @ weights)