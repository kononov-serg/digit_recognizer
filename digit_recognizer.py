import tensorflow as tf
import pandas as pd
#test comment:
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  x_batch = train_df.iloc[range(i, i + 10), range(1, train_df.shape[1])]
  y_batch = train_df.iloc[range(i, i + 10), 0]
  train_step.run(feed_dict={x: x_batch, y_: y_batch})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# ImageId, Label
