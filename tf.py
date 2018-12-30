import tensorflow as tf 
import numpy as np 
BATCH_SIZE = 8
seed = 7
learning_rate =0.08
steps = 10000000
#产生随机数
rng = np.random.RandomState(seed)
#产生矩阵随机数作为输入集
x = rng.rand(32,2)
#自定义标准答案
Yi = [[int(x0+x1<1)] for (x0,x1) in x]
#print(x,y_)
#d定义神经网络输入、参数、和输出、定义前向传播过程
X  = tf.placeholder(tf.float32,shape = (None,2))
y_ = tf.placeholder(tf.float32,shape = (None,1))

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1]))

a = tf.matmul(X,w1)
y = tf.matmul(a,w2)

#定义损失函数以及反向传播方法
#损失函数为均方差，优化方法为梯度下降
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.MomentumOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#会话以及定义训练的次数
with tf.Session() as sess:
    input_op = tf.global_variables_initializer()
    sess.run(input_op)
    print("为训练前的参数:",sess.run(w1),sess.run(w2))

    for i in range(steps):
        start = (i*BATCH_SIZE)%32
        end = min(start+BATCH_SIZE,32)
        sess.run(train_step,feed_dict= {X:x[start:end],y_:Yi[start:end]})

        if i %5000 ==0:
             #print("w1:",sess.run(w1),"w2:",sess.run(w2))
             loss_total = sess.run(loss,feed_dict={X:x,y_:Yi})
             print(loss_total)
    print("w1:","\n",sess.run(w1),"\n","w2:","\n",sess.run(w2))