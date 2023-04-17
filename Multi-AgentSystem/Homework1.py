import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0,1,50)

parameter = np.array([[2,4,3,5,0],[7,4,5,0,8]],dtype=np.float16)

r = np.zeros((p.shape[0],parameter.shape[1]))
for i in range(p.shape[0]):
    r[i,:] = (1-p[i]) * parameter[0] + p[i] * parameter[1]


plt.plot(p, r)
x1 = [0.0,1.0]
y1 = [5, 8]
x2 = [0.2, 0.4, 2/3]
y2 = [4, 4, 16/3]
plt.plot(p,np.max(r,axis=1),"k.")
plt.scatter(x1,y1,c='r')
plt.scatter(x2,y2,c='b')
plt.grid()
plt.xlabel("p:the probability of player I plays strategy B")
plt.ylabel("Expect payoffs of player II")
plt.legend(['a','b','c','d','e','BR'],ncol=2)
for i in range(len(x1)):
    plt.text(x1[i]-0.02,y1[i]-0.5,f'$P_{i+1}$')
for i in range(len(x2)):
    plt.text(x2[i]-0.02,y2[i]+0.3,f'$M_{i+1}$')
plt.show()