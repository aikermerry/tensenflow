import numpy as np 
import tensorflow as tf 

#设置图片数量、高度、宽度、颜色通道
x_shape = [1,4,4,1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32,shape=x_shape)

my_filter = tf.constant(0.25,shape = [2,2,1,1])
my_strides = [1,2,2,1]
mov_avg_layer = tf.nn.conv2d(x_data,my_filter,my_strides,
padding = 'SAME''',name = 'Moving_Avg_Window')

def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A =tf.constant([[1.,2.],[-1.,3.]])
    b = tf.constant(1.,shape = [2,2])
    temp1 = tf.matmul(A,input_matrix_sqeezed)
    temp = tf.add(temp1,b)
    return (tf.sigmoid(temp))
with tf.name_scope("custom_layer") as score:
    custom_layer1 = custom_layer(mov_avg_layer)
    print(tf.Session().run(custom_layer1,feed_dict={x_data:x_val}))