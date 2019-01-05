import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets
steps = 10000
bitch_size = 20
LEARNING_RATE_BASE = 0.02
LEARNING_RATE_DECAY=0.9999
regularizer=0.01
#导入数据
iris = datasets.load_iris()
out_target = np.array([[1. ]if x==0 else [0.] for x in iris.target])
in_data = np.array([[x[2],x[3]] for x in iris.data])
data_len = len(iris.data)
#创建变量

x_in = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape = (None,1))
global_step = tf.Variable(0,trainable=False)

w1 = tf.Variable(tf.random_normal(shape=[2,3]))
tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w1))

w2 = tf.Variable(tf.random_normal(shape= [3,1]))
tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w2))
a =tf.matmul(x_in,w1)
y_out = tf.matmul(a,w2)

#loss = tf.reduce_mean(tf.square(y_out-y_))
loss =tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out,labels=y_)
loss_total = loss+ tf.add_n(tf.get_collection("losses"))

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
    global_step,
    data_len/bitch_size,
    LEARNING_RATE_DECAY,
    staircase = True)

    

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total,
        global_step=global_step)


with tf.Session() as sess:

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(steps):
        START = i*bitch_size%data_len
        END = min(START +bitch_size,data_len)

        sess.run(train_step,feed_dict = {x_in:in_data[START:END],y_:out_target[START:END]})
        if i%500==0:
            print("loss:"+str(sess.run(loss_total,feed_dict = {x_in:in_data,y_:out_target})))

    print(sess.run(w1))

    print(sess.run(y_out,feed_dict={x_in:in_data}))