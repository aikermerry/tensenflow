import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets

steps = 10000
batch_size = 20

#导入数据
iris = datasets.load_iris()
y_target = np.array([1. if x==0 else 0. for x in iris.target])
in_data = np.array([[x[2],x[3]] for x in iris.data])
data_len = len(iris.data)
#定义线性模型：x1 = x2 *A+ b
x1_data =tf.placeholder(tf.float32,shape = [None,1])
x2_data =tf.placeholder(tf.float32,shape = [None,1])
y_ =tf.placeholder(tf.float32,shape = [None,1])

A = tf.Variable(tf.random_normal(shape= (1,1)))
b= tf.Variable(tf.random_normal(shape = (1,1)))

my_mult = tf.matmul(x2_data,A)
my_add = tf.add(my_mult,b)
my_out = tf.subtract(x1_data,my_add)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_out,labels=y_)

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(xentropy)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(1000):
        rand_index = np.random.choice(data_len,size=batch_size)
        rand_x = in_data[rand_index]
        rand_x1 = np.array([[x[0]] for x in rand_x])
        rand_x2 = np.array([[x[1]] for x in rand_x])
        rand_y = np.array([[y] for y in y_target[rand_index]])
        sess.run(train_step,feed_dict={x1_data:rand_x1,x2_data:rand_x2,y_:rand_y})
        
    
    [[slope]]=sess.run(A)
    [[intercept]]=sess.run(b)
    x = np.linspace(0,3,num=50)
    ablineValues=[]
    for i in x:
        ablineValues.append(slope*i+intercept)
    set_x = [a[1] for i,a,in enumerate(in_data) if y_target[i] ==1]
    swt_y = [a[0] for i,a,in enumerate(in_data) if y_target[i] ==1]
    no_x = [a[1] for i,a,in enumerate(in_data) if y_target[i] ==0]
    no_y = [a[0] for i,a,in enumerate(in_data) if y_target[i] ==0]

    plt.plot(set_x,swt_y,"rs",label='setosa')
    plt.plot(no_x,no_y,"g^",label='unsetosa')
    plt.plot(x,ablineValues,'b-')
    plt.xlim([0.0,2.7])
    plt.ylim([0.0,7.1])
    plt.suptitle('linear\' Separator For I.setosa',fontsize=20)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='lower right')
    plt.show()








