import tensorflow as tf

def preprocess(inputs, batch_norm):
  pre = tf.layers.conv2d(inputs=inputs,filters=64,kernel_size=(7,7),strides=(2,2),padding="same",activation=None,kernel_regularizer=None)
  pre = tf.layers.batch_normalization(pre,fused=True,training=batch_norm)
  pre = tf.nn.relu(pre)
  pre = tf.layers.max_pooling2d(inputs=pre,pool_size=(3,3),strides=(2,2),padding='same')
  return pre

def postprocess(inputs,batch_norm):
  post = tf.layers.conv2d_transpose(inputs=inputs,filters=32,kernel_size=(3,3),strides=(2,2),padding="same",activation=None,kernel_regularizer=None)
  post = tf.layers.batch_normalization(post,fused=True,training=batch_norm)
  post = tf.nn.relu(post)
  post = tf.layers.conv2d(inputs=post,filters=32,kernel_size=(3,3),strides=(1,1),padding="same",activation=None,kernel_regularizer=None)
  post = tf.layers.batch_normalization(post,fused=True,training=batch_norm)
  post = tf.nn.relu(post)
  post = tf.layers.conv2d_transpose(inputs=post,filters=20,kernel_size=(2,2),strides=(2,2),padding="same",activation=None,kernel_regularizer=None)
  return post

def fullconv(inputs,filters,kernel_size,strides,regularizer,batch_norm) :
  conved = tf.layers.conv2d_transpose(inputs=inputs,filters=filters,kernel_size=kernel_size,strides=strides,padding="same",activation=None,kernel_regularizer=None)
  conved = bn(conved,is_training=batch_norm)
  return conved

def encoder(inputs,m,n,batch_norm) :
  conv1 = tf.layers.conv2d(inputs=inputs,filters=n,kernel_size=(3,3),strides=(2,2),padding="same",activation=None,kernel_regularizer=None)
  conv1 = tf.layers.batch_normalization(conv1,fused=True,training=batch_norm)
  conv1 = tf.nn.relu(conv1)
  conv1 = tf.layers.conv2d(inputs=conv1,filters=n,kernel_size=(3,3),strides=(1,1),padding="same",activation=None,kernel_regularizer=None)
  conv1 = tf.layers.batch_normalization(conv1,fused=True,training=batch_norm)
  shortcut1 = tf.layers.conv2d(inputs=inputs,filters=n,kernel_size=(3,3),strides=(2,2),padding="same",activation=None,kernel_regularizer=None)
  shortcut1 = tf.layers.batch_normalization(shortcut1,fused=True,training=batch_norm)
  conv1 += shortcut1
  conv1 = tf.nn.relu(conv1)
  conv2 = tf.layers.conv2d(inputs=conv1,filters=n,kernel_size=(3,3),strides=(1,1),padding="same",activation=None,kernel_regularizer=None)
  conv2 = tf.layers.batch_normalization(conv2,fused=True,training=batch_norm)
  conv2 = tf.nn.relu(conv2)
  conv2 = tf.layers.conv2d(inputs=conv2,filters=n,kernel_size=(3,3),strides=(1,1),padding="same",activation=None,kernel_regularizer=None)
  conv2 = tf.layers.batch_normalization(conv2,fused=True,training=batch_norm)
  conv2 += conv1
  conv2 = tf.nn.relu(conv2)
  return conv2

def decoder(inputs,m,n,batch_norm) :
  conved = tf.layers.conv2d(inputs=inputs,filters=m//4,kernel_size=(1,1),strides=(1,1),padding="same",activation=None,kernel_regularizer=None)
  conved = tf.layers.batch_normalization(conved,fused=True,training=batch_norm)
  conved = tf.nn.relu(conved)
  conved = tf.layers.conv2d_transpose(inputs=conved,filters=m//4,kernel_size=(3,3),strides=(2,2),padding="same",activation=None,kernel_regularizer=None)
  conved = tf.layers.batch_normalization(conved,fused=True,training=batch_norm)
  conved = tf.nn.relu(conved)
  conved = tf.layers.conv2d(inputs=conved,filters=n,kernel_size=(1,1),strides=(1,1),padding="same",activation=None,kernel_regularizer=None)
  conved = tf.layers.batch_normalization(conved,fused=True,training=batch_norm)
  conved = tf.nn.relu(conved)
  return conved

def linknet_model(inputs, is_training):
  pre = preprocess(inputs, batch_norm=is_training)
  enc1 = encoder(pre,m=64,n=64,batch_norm=is_training)
  enc2 = encoder(enc1,m=64,n=128,batch_norm=is_training)
  enc3 = encoder(enc2,m=128,n=256,batch_norm=is_training)
  enc4 = encoder(enc3,m=256,n=512,batch_norm=is_training)
  dec1 = decoder(enc4,m=512,n=256,batch_norm=is_training)
  dec1 += enc3
  dec2 = decoder(dec1,m=256,n=128,batch_norm=is_training)
  dec2 += enc2
  dec3 = decoder(dec2,m=128,n=64,batch_norm=is_training)
  dec3 += enc1
  dec4 = decoder(dec3,m=64,n=64,batch_norm=is_training)
  outputs = postprocess(dec4, batch_norm=is_training)
  return outputs