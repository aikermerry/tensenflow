# 手写数字识别

1. 导入数据：

   ```
   from tensorflow.examples.tutorials.mnist import input_data
   mnist = input_data.read_data_sets('./data/',one_hot =True)
   
   如果没找到将会在这些地方去发现
   Extracting ./data/train-images-idx3-ubyte.gz#用来训练的数据
   Extracting ./data/train-labels-idx1-ubyte.gz
   Extracting ./data/t10k-images-idx3-ubyte.gz#用来测试的数据
   Extracting ./data/t10k-labels-idx1-ubyte.gz
   
   ```

   

2. 降维度：将每一个像素点作为一个输入特征

3. 常用函数：

   ```
   tf.get_collection("")#从集合中去除全部变量，生成一个列表
   tf.add_n([]) #将列表内对应元素相加
   tf.cast(x,dtype)#把x 转化为特定类型
   tf.argmax(x,axis)#返回最大值所在的下标
   os.path.join(“home”,'name')#返回home/name
   ```

   

4. 保存模型

   ```
   saver = tf.train.Saver()
   saver.save(sess,os.path.join("模型保存路径"，“模型名字”),global_step=global_step)
   ```

   

5. 还原模型

    

   ```
   with tf.Session() as sess:
   	ckpt = tf.train.get_checkpoint_state(“模型路径”)
   	if ckpt and ckpt.mpdel_check_path:
   		saber.restore(sess,ckpt.model_checkpoint_path)
   ```

6. 实例化还原滑动平均值的saver

   ```
   ema = tf.train.ExponentialMovingAverage(滑动平均基数)
   ema_restore = ema.variables_to_restore()
   saver = tf.train.Saver(ema_restore)
   ```

   

7. 准确率计算
   将预测值与标准值比对.

   ```
   correct_pre = tf.equal(tf.argmax(y,1),tf.argmax(y_1))
   accuracy = tf.reduce_mean(tf.cost(correct_pre,tf.float32))
   
   ```

