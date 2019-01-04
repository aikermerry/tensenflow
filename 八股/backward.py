#反向传播过程
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import forward
import generateds

STEPS = 400000
BATCH_SIZE = 30
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.99
regularizer = 0.01
data_size = 300


def backward():

    X,Y_,Y_c = generateds.generateds()

    x = tf.placeholder(tf.float32,shape=(None,2))
    y_ = tf.placeholder(tf.float32,shape=(None,1))

    y = forward.forward(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
#定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse+tf.add_n(tf.get_collection('losses'))
    #ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,\
     #   labels=tf.argmax(y_,1))
    #y_y_ = tf.reduce_mean(ce)
    #加入正则化后
    #loss_total = y_y_ + tf.add_n(tf.get_collection("losses"))
   
#指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
        global_step,
        data_size/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total,
        global_step=global_step)

    #滑动平均
    #ema = tf.train.ExponentialMovingAvrage(MOVING_AVERAGE_DECAY,gloabl_step)
    #ema_OP = ema.apply(tf.trainable_variables())
    #with tf.control_dependencies([train_step,ema_op]):
    #    train_op = tf.no_op(name = 'train')
    

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = ((i*BATCH_SIZE)+BATCH_SIZE)%data_size
            end = min(start+BATCH_SIZE,data_size)
            sess.run(train_step,feed_dict={x:X[start:end]  ,y_:Y_[start:end]})

            if i%2000 ==0 :
                loss_v=sess.run(loss_total,feed_dict={x:X,y_:Y_})
                print(i,loss_v)
              

        xx,yy = np.mgrid[-3:3:0.01,-3:3:0.01]
        grid = np.c_[xx.ravel(),yy.ravel()]
        probs=sess.run(y,feed_dict = {x:grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

if __name__ == "__main__":
    backward()


