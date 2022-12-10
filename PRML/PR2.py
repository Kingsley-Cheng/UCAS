import numpy as np
num = 1000
alpha=0.9
e=0.9
b = np.array([1,2,3])
A = np.array([[3/4,1/4,0],[1/4,3/4-e, e],[0,e,1-e]])
j = np.ones(3)
for i in range(num):
    j = b + alpha * np.dot(A,j)
    print(i)
    print(j)