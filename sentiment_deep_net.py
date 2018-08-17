#!/usr/bin/python3

import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels

X = tf.placeholder('float')
y = tf.placeholder('float')

class deep_net :
    def __init__(self) :
        self.train_x, self.train_y, self.test_x, self.test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

        self.n_nodes_hl1 = 1500
        self.n_nodes_hl2 = 1500
        self.n_nodes_hl3 = 1500

        self.n_classes = 2
        self.batch_size = 100

    def neural_network_model(self, data) :
        hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([len(self.train_x[0]), self.n_nodes_hl1])), 'biases' : tf.Variable(tf.random_normal([self.n_nodes_hl1]))}
        hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])), 'biases' : tf.Variable(tf.random_normal([self.n_nodes_hl2]))}
        hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])), 'biases' : tf.Variable(tf.random_normal([self.n_nodes_hl3]))}
        outer_layer = {'weights' : tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])), 'biases' : tf.Variable(tf.random_normal([self.n_classes]))}

        l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.add(tf.matmul(l3, outer_layer['weights']), outer_layer['biases'])

        return output

    def train_neural_network(self, X) :
        prediction = self.neural_network_model(X)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        hm_epoch = 9
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epoch) :
                epoch_loss = 0
                i = 0
                if i < len(self.train_x) :
                    start = i
                    end = i + self.batch_size
                    batch_x = self.train_x[start:end]
                    batch_y = self.train_y[start:end]

                    _, c = sess.run([optimizer, cost], feed_dict = {X : batch_x, y : batch_y})
                    epoch_loss += c

                    i += self.batch_size

                print('Epoch ', epoch, ' completed out of', hm_epoch, ' loss : ', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy : ', accuracy.eval({X : self.test_x, y : self.test_y}))

clf = deep_net()
clf.train_neural_network(X)
