import numpy as np
import tensorflow as tf
from model import linknet_model
from make_image_cityscapes import make_img_cityscapes

class Linknet:
  def __init__(self,height,width,num_categories):
    self.sess = tf.InteractiveSession()
    self.inputs = tf.placeholder(tf.float32, [None, height, width, 3])
    self.labels = tf.placeholder(tf.float32, [None, height, width, num_categories])
    self.is_training = tf.placeholder(tf.bool)
    self.outputs = linknet_model(self.inputs,self.is_training)
    self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,logits=self.outputs))
    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.control_dependencies(self.update_ops):
      self.train_step = tf.train.AdamOptimizer(5e-5).minimize(self.cross_entropy)
    self.correct_prediction = tf.equal(tf.argmax(self.outputs, 3), tf.argmax(self.labels, 3))
    self.output_data = tf.argmax(self.outputs, 3)
    self.accurate_data = tf.argmax(self.labels, 3)
    self.correct_prediction = tf.equal(self.output_data, self.accurate_data)
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    self.sess.run(tf.global_variables_initializer())
    self.file_count = 0
    self.height = height
    self.width = width
    self.num_categories = num_categories

  def train(self,x_train_base,y_train_base,batch_size):
    train_size = len(x_train_base)
    for batch_count in range(train_size // batch_size):
      if(batch_count % 10 == 0):
        print("batch:",batch_count)
      x_train = self.x_train_base[batch_size*batch_count:batch_size*(batch_count+1)]
      y_train = self.y_train_base[batch_size*batch_count:batch_size*(batch_count+1)]
      self.sess.run(self.train_step,feed_dict={self.inputs: x_train, self.labels: y_train, self.is_training: True})
  
  def test(self,x_test,y_test):
    output_data2 = np.array(self.sess.run(self.output_data,feed_dict={self.inputs: x_test, self.labels: y_test, self.is_training: False}), dtype="uint8")
    accurate_data2 = np.array(self.sess.run(self.accurate_data,feed_dict={self.inputs: x_test, self.labels: y_test, self.is_training: False}), dtype="uint8")
    IOU = np.zeros(self.num_categories)
    accuracy = 0.0
    iou_count = np.zeros(self.num_categories)
    for j in range(self.num_categories) :
      for i in range(output_data2.shape[0]) :
        TP = np.sum(np.logical_and(np.equal(output_data2[i], j), np.equal(accurate_data2[i],j)))
        GT = np.sum(np.equal(accurate_data2[i],j))
        PR = np.sum(np.equal(output_data2[i],j))
        if(GT + PR - TP > 0):
          IOU[j] += TP / (GT + PR - TP)
          iou_count[j] += 1
          accuracy += TP
      if(iou_count[j] > 0):
        IOU[j] /= iou_count[j]
    iou = np.sum(IOU) / np.sum(iou_count > 0)
    accuracy /= (output_data2.shape[0]*output_data2.shape[1]*output_data2.shape[2])

    print("accuracy",accuracy)
    print("IOU:",iou)
    for i in range(np.min((output_data2.shape[0],10))) :
      self.file_count += 1
      make_img_cityscapes(output_data2[i],self.file_count,self.height,self.width)
    print("save_image")


