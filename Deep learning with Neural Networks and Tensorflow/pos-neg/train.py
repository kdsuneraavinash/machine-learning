# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 08:10:10 2019
edited from ../mnist/mnist.py

@author: Sunera
"""

import tensorflow as tf
import feature_set_gen
import pickle

# (Because of Spyder) ~ Reset tensorflow
tf.reset_default_graph()

# Taking input data set
with open("save.pickle", "rb") as fb:
    pos_neg = pickle.load(fb)
k_train_x, k_train_y, k_test_x, k_test_y = pos_neg

input_size = len(k_train_x[0])

# Important params
input_nodes = input_size
hidden_layer_1_nodes = 5000
hidden_layer_2_nodes = 5000
hidden_layer_3_nodes = 5000
output_nodes = 2

batch_size = 100


# NN Model
def neural_network(input_layer):

    # Setting up all weights and biases
    hidden_layer_1 = {'weights': tf.get_variable("Hidden_Layer_1_Weights", shape=[input_nodes, hidden_layer_1_nodes]),
                      'biases': tf.get_variable('Hidden_Layer_1_Biases', shape=[1, hidden_layer_1_nodes])}
    hidden_layer_2 = {'weights': tf.get_variable("Hidden_Layer_2_Weights", shape=[hidden_layer_1_nodes, hidden_layer_2_nodes]),
                      'biases': tf.get_variable('Hidden_Layer_2_Biases', shape=[1, hidden_layer_2_nodes])}
    hidden_layer_3 = {'weights': tf.get_variable("Hidden_Layer_3_Weights", shape=[hidden_layer_2_nodes, hidden_layer_3_nodes]),
                      'biases': tf.get_variable('Hidden_Layer_3_Biases', shape=[1, hidden_layer_3_nodes])}
    output_layer = {'weights': tf.get_variable("Output_Layer_Weights", shape=[hidden_layer_3_nodes, output_nodes]),
                    'biases': tf.get_variable('Output_Layer_Biases', shape=[1, output_nodes])}

    # Netword model
    hidden_layer_1_weighted = tf.matmul(input_layer, hidden_layer_1['weights'])
    hidden_layer_1_biased = tf.add(
        hidden_layer_1_weighted, hidden_layer_1['biases'])
    hidden_layer_1_output = tf.nn.relu(hidden_layer_1_biased)

    hidden_layer_2_weighted = tf.matmul(
        hidden_layer_1_output, hidden_layer_2['weights'])
    hidden_layer_2_biased = tf.add(
        hidden_layer_2_weighted, hidden_layer_2['biases'])
    hidden_layer_2_output = tf.nn.relu(hidden_layer_2_biased)

    hidden_layer_3_weighted = tf.matmul(
        hidden_layer_2_output, hidden_layer_3['weights'])
    hidden_layer_3_biased = tf.add(
        hidden_layer_3_weighted, hidden_layer_3['biases'])
    hidden_layer_3_output = tf.nn.relu(hidden_layer_3_biased)

    output_layer_weighted = tf.matmul(
        hidden_layer_3_output, output_layer['weights'])
    output = tf.add(output_layer_weighted, output_layer['biases'])

    return output


def train_neural_network(n_epoch):
    # X and Y placeholders (input and expected output)
    x = tf.placeholder(tf.float32, shape=[None, input_nodes], name="X")
    y = tf.placeholder(tf.float32, shape=[None, output_nodes], name="Y")

    # Prediction tensor using model
    prediction = neural_network(x)

    # Copute cost function
    total_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prediction, labels=y)
    cost = tf.reduce_mean(total_entropy)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Start Training
    with tf.Session() as session:
        # Initialize all variables
        session.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):

            # initial epoch loss
            epoch_loss = 0

            for train_x, train_y in feature_set_gen.next_batch(k_train_x, k_train_y, batch_size):
                # Train
                _, batch_cost = session.run([optimizer, cost], feed_dict={
                                            x: train_x, y: train_y})
                epoch_loss += batch_cost

            # Pront progress
            print(
                "epoch completed {:>2}/{:>2} ---- Epoch cost {:.5f}".format(epoch + 1, n_epoch, epoch_loss))

        # Take max from prediction and expected ouput and check if they are same
        is_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # Define and calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        accuracy_digit = accuracy.eval({x: k_test_x, y: k_test_y})

    print("Model trained... Accuracy = {}".format(accuracy_digit))


if __name__ == '__main__':
    train_neural_network(100)
