import tensorflow as tf 
import matplotlib.pyplot as plt 

#创建预测序列和目标张量
x_vals = tf.linspace(-1.,1.,500)
target = tf.constant(0.)
sess = tf.Session()

#L2正则损失表达
l2_y_vals = tf.square(target-x_vals)
l2_y_out = sess.run(l2_y_vals)

#L1损失函数
l1_y_vals = tf.abs(target-x_vals)
l1_y_out = sess.run(l1_y_vals)

#Psendo_Huber 损失函数
delta1 = tf.constant([[0.25]])
phuber1_y_vals = tf.matmul(tf.square(delta1),
    tf.sqrt(1.+tf.square((target - x_vals)/delta1))-1.)
phuber1_y_out = sess.run(phuber1_y_vals)

deltal2 = tf.constant([[5.]])
phuber2_y_vals = tf.matmul(tf.square(deltal2),tf.sqrt(1.+tf.square((target -x_vals)/deltal2))-1.)
phuber2_y_out = sess.run(phuber2_y_vals)
#######################################
x_vals = tf.linspace(-3.,5.,500)
target = tf.constant(1.)
targets = tf.fill([500,],1.)

#Hinge损失函数
#hinge_y_vals = tf.maximum(0.,1.,-tf.multiply(target,x_vals))
#hinge_y_out = tf.run(hinge_y_vals)

#Cross_entropy loss 交叉熵损失函数

#xentropy_y_vals = tf.matmul(target,tf.log(x_vals)) -tf.matmul((1. -target),tf.log(1.-x_vals))
#xentropy_y_out = sess.run(xentropy_y_vals)

#Sigmoid交叉熵损失函数

#xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(x_vals,targets)
#xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

#加权交叉熵损失函数

weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals,targets,weight)
xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

#Softmax交叉熵损失函数

unscaled_logits = tf.constant([[1.,-3.,10.]])
target_dist = tf.constant([[0.1,0.02,0.88]])
softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(unscaled_logits,target_dist)
print(sess.run(softmax_xentropy))

#稀疏Softmax交叉熵损失函数

unscaled_logits = tf.constant([1.,-3.,10.])
sparse_ta





rget_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(unscaled_logits,sparse_target_dist)
print(sess.run(sparse_xentropy))
