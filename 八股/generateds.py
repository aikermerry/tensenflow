#coding:utf-8
import numpy as np 
import matplotlib.pyplot as plt 
seed =0

def generateds():

#生成随机数
    rdm = np.random.RandomState(seed)
    X = rdm.randn(300,2)
    Y_ = [int(x0*x0+x1*x1<2.0) for (x0,x1) in X]

    y_c = [['red' if y else 'blue'] for y in Y_]

    X = np.vstack(X).reshape(-1,2)
    Y_ = np.vstack(Y_).reshape(-1,1)

    return X,Y_,y_c

if __name__ == "__main__":
    X,Y_,y_c=generateds()
    print(max(X[:,0]))
    plt.scatter(X[:,0],X[:,1],c = np.squeeze(y_c))
    plt.show()
