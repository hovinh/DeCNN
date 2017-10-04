import numpy as np
import tensorflow as tf
from scipy import misc
import tf_cnnvis
import time


'''DEFINING GRAPH'''

graph = tf.Graph()
with graph.as_default():
    #last dimension muse be specified
    x = tf.placeholder(tf.float32, shape=(None, None, None, 3)) 
    
    pool_1 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                            padding='SAME', name='pool1')
    
    pool_2 = tf.nn.max_pool(pool_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                            padding='SAME', name='pool2')
    
    pool_3 = tf.nn.max_pool(pool_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                            padding='SAME', name='pool3')
    
    pool_4 = tf.nn.max_pool(pool_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                            padding='SAME', name='pool4')
    
    pool_5 = tf.nn.max_pool(pool_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                            padding='SAME', name='pool5')


'''EXPERIMENT'''

image = misc.imread('Lenna.png')
image = image.astype(np.float32)
image = [image]

layers = ['p'] # only show pooling layer

start = time.time()

is_success = tf_cnnvis.activation_visualization(graph_or_path = graph, value_feed_dict = {x : image}, 
                                      layers=layers, path_logdir="./Log/Pooling", path_outdir="./Output/Pooling")

is_success = tf_cnnvis.deconv_visualization(graph_or_path = graph, value_feed_dict = {x : image},
                                  layers=layers, path_logdir="./Log/Pooling", path_outdir="./Output/Pooling")
start = time.time() - start
print("Total Time = %f" % (start))
