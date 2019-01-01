import tensorflow as tf
import numpy as np
ma_array = np.array([[1.,3.,5.,7.,9.],
    [-2., 0., 2., 4., 6.],[-6., -3., 0., 3.,6.]])

x_vals = np.array([ma_array,ma_array+1])
xdata = tf.placeholder(tf.float32,shape=(3,5))

w1 =tf.constant([[1.],[0.],[-1.],[2.],[4.]])
w2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

prod1 = tf.matmul(xdata,w1)
prod2 = tf.matmul(prod1,w2)
add1 = tf.add(prod2,a1)

with tf.Session() as sess:
    print(sess.run(add1,feed_dict={xdata:x_vals[1]}))