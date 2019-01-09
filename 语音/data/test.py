import matplotlib.pyplot as plt 
import scipy.io as scio

data = scio.loadmat('./output.mat')
m=[]

for i in data :

    m.append(i)




in_data = data[m[5]]
print(len(m))

print(in_data.shape)

#plt.plot(in_data[1])

#plt.show()
