def test():

    with tf.Graph().as_default() as g:
        #定义x y_ y
        #实例化可还原滑动平均值的server
        #计算正确率
        with tf.Session() as sess:
         #加载模型 
            ckpt = tf.train.get_checkpoint_state(“模型路径”)

            if ckpt and ckpt.mpdel_check_path:
                saber.restore(sess,ckpt.model_checkpoint_path)
                #恢复会话轮数
                global_step = ckpt.model_checkpoint_split('/')[-1].split('-')[-1]
                #计算准确率
                accuracy_score = sess.run(accuracy,feed_dict={})
                print("after %s train steps,test accuracy = %g "%(global_step,accuracy_score))
            else :
                print("没有找到模型")
                 