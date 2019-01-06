import time
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward



def test():

    with tf.Graph().as_default() as g:

        #定义x y_ y
        x = tf.placeholder(tf.float32,shape=(None,forward.INPUT_NODE))
        y_ = tf.placeholder(tf.float32,shape=(None,forward.OUT_NODE))


        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()  
        saver = tf.train.Saver(ema_restore)


        correct_pre = tf.equal(tf.argmax(y,1),tf.argmax(y_1))
        accuracy = tf.reduce_mean(tf.cost(correct_pre,tf.float32))
        #实例化可还原滑动平均值的server
        #计算正确率
        with tf.Session() as sess:
         #加载模型 
            ckpt = tf.train.get_checkpoint_state(backwward.mode_path)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                #恢复会话轮数
                global_step = ckpt.model_checkpoint_split('/')[-1].split('-')[-1]
                #计算准确率
                accuracy_score = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                print("after %s train steps,test accuracy = %g "%(global_step,accuracy_score))
            else :
                print("没有找到模型")


if __name__== "__mian__":

    mnist=input_data.read_data_sets("/usr/local/lib/python3.6/dist-packages/tensorflow/contrib/learn/python/learn/datasets/data/",one_hot = True)
    test(mnist)
                 