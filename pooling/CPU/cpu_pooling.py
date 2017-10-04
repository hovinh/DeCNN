import numpy as np
import tensorflow as tf
from scipy import misc
from PIL import Image


'''DEFINING GRAPH'''

x = tf.placeholder(tf.float32, shape=(1, None, None, None))

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
 
def unpool(x, size):
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * size, sh[2] * size, sh[3]]
        return tf.reshape(out, out_size)
    
    shv = tf.shape(x); print (sh); print (shv); print (sh[3])
    ret = tf.reshape(out, tf.stack([-1, shv[1] * size, shv[2] * size, 3]))
    ret.set_shape([None, None, None, 3])
    return ret
    
unpool_5 = unpool(pool_5, 2)

unpool_4 = unpool(unpool_5, 2)

unpool_3 = unpool(unpool_4, 2)

unpool_2 = unpool(unpool_3, 2)

unpool_1 = unpool(unpool_2, 2)

    
'''EXPERIMENT'''

image = misc.imread('Lenna.png')
image = image.astype(np.float32)
print (image.shape)


session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

tests = [
    pool_1.eval(feed_dict={x: [image]}),
    pool_2.eval(feed_dict={x: [image]}),
    pool_3.eval(feed_dict={x: [image]}),
    pool_4.eval(feed_dict={x: [image]}),
    pool_5.eval(feed_dict={x: [image]}),
    unpool_5.eval(feed_dict={x: [image]}),
    unpool_4.eval(feed_dict={x: [image]}),
    unpool_3.eval(feed_dict={x: [image]}),
    unpool_2.eval(feed_dict={x: [image]}),
    unpool_1.eval(feed_dict={x: [image]})
]

session.close()

names = ['pool_1', 'pool_2', 'pool_3', 'pool_4', 'pool_5', 'unpool_5', 'unpool_4', 'unpool_3', 'unpool_2', 'unpool_1']

for idx in range(len(tests)):
    name = names[idx]; test = tests[idx]
    im =Image.fromarray(np.uint8(test[0]), 'RGB')
    im.save(name + '.png')

