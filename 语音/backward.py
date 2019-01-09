#反向传播过程
import tensorflow as tf 
import numpy as np  
import forward
import os 
import scipy.io as scio
data_out = scio.loadmat('./data/output.mat')
data_in = scio.loadmat('./data/intput.mat')

STEPS = 40000
BATCH_SIZE = 50
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.9999
MOVING_AVERAGE_DECAY =0.99
regularizer = 0.0001
mode_path = "./model"
mode_name = "noise_model"

def backward(data_size,all_out,all_in):
    x = tf.placeholder(tf.float32,shape=(None,forward.INPUT_NODE))
    y_ = tf.placeholder(tf.float32,shape=(None,forward.OUT_NODE))

    y,w1,w2 = forward.forward(x,regularizer)
    global_step = tf.Variable(tf.constant(0),trainable=False)
#定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse+tf.add_n(tf.get_collection('losses'))
    #ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
    #loss = tf.reduce_mean(ce)
    #加入正则化后
    #loss_total = loss + tf.add_n(tf.get_collection("losses"))
   
#指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
        global_step,
        data_size/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)

    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total,
     #   global_step=global_step)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total,
     global_step=global_step)
    #滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name = 'train')
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(mode_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            

        for i in range(40000):
            try:
                START = ((i*BATCH_SIZE)+BATCH_SIZE)%data_size
                END = min(START+BATCH_SIZE,data_size)

                _,loss_value,step=sess.run([train_op,loss_total,global_step],
                    feed_dict={x:all_in[START:END] ,y_:all_out[START:END]})

                if i%2000 ==0 :
                    #print(str(sess.run(y,feed_dict={x:xs ,y_:ys})))
                    print("--"*20)

                    print("steps:"+str(step)+" 损失："+str(loss_value))
                    #保存
                    saver.save(sess,os.path.join(mode_path,mode_name),global_step=global_step)
            except Exception as f:
                print("**"*50)
                print("不定义捕获的异常是：%s"%f)
            finally:
                saver.save(sess,os.path.join(mode_path,mode_name),global_step=global_step)
           
              

if __name__ == "__main__":

    save_name1=[]
    save_name2=[]
    all_in=[]
    all_out=[]
    for i in data_in :
        save_name1.append(i)

    for i in data_out :
        save_name2.append(i)
    for i in range(129):
        if i>=3:
            for x in data_in[save_name1[i]]:
                all_in.append(x)
            for y in data_out[save_name2[i]]:
                all_out.append(y)

    del save_name2 
    del save_name1 
    del data_in 
    del data_out
    print(len(all_out))
    try :
       
        backward(len(all_out),all_out,all_in)
    except Exception as f:
        print("发生了错误，错误是：s%"%f)

