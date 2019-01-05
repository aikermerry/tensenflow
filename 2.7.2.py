import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

bitch_size = 20

x_in = np.random.normal(1,0.1,100)
y_ = np.repeat(10.,100)

X =tf.placeholder(tf.float32,shape = (None,1))
Y_ = tf.placeholder(tf.float32,shape = (None,1))

w1 = tf.Variable(tf.random_normal(shape=[1,1]))

y = tf.matmul(X,w1)

loss = tf.reduce_mean(tf.square(y-Y_))

steps = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    input_op = tf.global_variables_initializer()
    sess.run(input_op)
    loss_stochastic = []
    for i in range(40000):
       #starts = (i*bitch_size)%100
        #ends = min(starts+bitch_size,100)
        rand_index = np.random.choice(100, size=bitch_size)
        rand_x = np. transpose ( [x_in[rand_index]])
        rand_y = np. transpose ( [y_ [rand_index]])

        sess.run(steps,feed_dict = {X:rand_x,Y_:rand_y})
        if i%2000==0:
            print ('Steps'+ str(i+1) +'A ='+ str(sess.run(w1)))
            tem_loss=sess.run(loss, feed_dict={X:rand_x,Y_:rand_y})
            print("loss:"+str(tem_loss))
            loss_stochastic.append(tem_loss)
    plt.plot(range(0, 100, 5), loss_stochastic,'b-', label='Stochastic_Loss')
    #plt.plot(range(0, 100, 5), loss_batch,'r - -', label='Batch’ Loss ,size=20')
    plt.show()



