import numpy as np
import tensorflow as tf 
x_in = np.array([1.,3.,5.,7.,9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant([[3.]])
my_product = tf.mul(x_data,m_const)
sess = tf.Session()
for x_datas in x_in:
    print(sess.run(my_product,feed_dict={x_data:x_datas}))