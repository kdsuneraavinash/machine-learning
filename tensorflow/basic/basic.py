# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:50:40 2018

@author: Sunera
"""

import tensorflow as tf

a = tf.constant(5)
b = tf.constant(10)

c1 = a*b
c2 = tf.multiply(a, b)


x1 = tf.constant([1, 2, 3, 4], shape=[2, 2])
x2 = tf.constant([1, 0, 0, 1], shape=[2, 2])

y1 = x1*x2
y2 = tf.multiply(x1, x2)
y3 = tf.matmul(x1, x2)

with tf.Session() as sess:
    print("a*b = \n{}\n".format(sess.run(c1)))
    print("tf.multiply(a, b) = \n{}\n".format(sess.run(c2)))
    
    print("x1*x2 = \n{}\n".format(sess.run(y1)))
    print("tf.multiply(x1, x2) = \n{}\n".format(sess.run(y2)))
    print("tf.matmul(x1, x2) = \n{}\n".format(sess.run(y3)))
    
    