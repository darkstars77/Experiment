import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#weight initailization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name='weights')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name='bias')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='max_pool')


x = tf.placeholder(tf.float32, shape=[None, 784], name='input_Img') #mnist date
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='input_GroundTruth') #ground truth

#1 conv layer
with tf.name_scope('1_conv_pool_layer') as scope:

    W_conv1 = weight_variable([5, 5, 1, 32]) #filter size, input channel, output channel
    b_conv1 = bias_variable([32])

    x_images = tf.reshape(x,[-1,28,28,1]) #img cnt, weight, height, channel
    h1_conv = conv2d(x_images,W_conv1)+b_conv1
    h1_pool = max_pool_2x2(h1_conv)


    #h1_pool (?,14,14,32) 

#2 conv layer
with tf.name_scope('2_conv_pool_layer') as scope:
    W_conv2 = weight_variable([5, 5, 32, 64]) #filter size, input channel, output channel
    b_conv2 = bias_variable([64])

    h2_conv = conv2d(h1_pool,W_conv2) + b_conv2
    h2_pool = max_pool_2x2(h2_conv)

    #h2_pool(?,7,7,64)

    
#fc1 layer
with tf.name_scope('1_FC_layer') as scope:
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    fc1_layer = tf.reshape(h2_pool,[-1,7*7*64])

    fc1_layer_output = tf.matmul(fc1_layer,W_fc1) + b_fc1

#fc2 layer
with tf.name_scope('2_FC_layer') as scope:
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    fc2_layer_output = tf.matmul(fc1_layer_output,W_fc2) + b_fc2
    y = fc2_layer_output

#train and evaluate model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
train_step=tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cross_entropy)

tf.summary.scalar('Loss',cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1)),tf.float32))
tf.summary.scalar('Accuracy',accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer=tf.summary.FileWriter('./summaryDir2/', sess.graph)
    sess.run(tf.global_variables_initializer())
   
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(50)
        summary, _, trainaccuracy = sess.run([merged ,train_step, accuracy], feed_dict={x:batch_x, y_:batch_y})
        if i%10==0 :
            print('Step:%d train Accuracy : %g' % (i,trainaccuracy))
            writer.add_summary(summary,i)
            
    
    print('\---Test Accuracy---\n')
    testaccuracy = sess.run(accuracy, feed_dict = {x:mnist.test.images , y_:mnist.test.labels})
    print('test Accuracy : %g' %(testaccuracy))




