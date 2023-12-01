from __future__ import print_function

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from methods.Python.PARTY.supporting_files.helpers import optimize


def getSparcityPrior(inputX, C_init=None, lambda1=0.01, lambda2=10000, optimizer='Adam', epochs=10000, learning_rate=0.1, print_step=50):
    tf.reset_default_graph()

    n_feat, n_sample = inputX.shape

    X = tf.placeholder(dtype=tf.float32, shape=[n_feat, n_sample], name='X')

    if C_init is None:
        C = tf.Variable(tf.random_uniform([n_sample, n_sample], -1, 1), name='C')
    else:
        C = tf.Variable(C_init, name='C')

    loss = X - tf.matmul(X, C)
    loss = tf.reduce_mean(tf.square(loss))

    # Create sparseness in C
    reg_lossC = tf.reduce_mean(abs(C))  # L1 loss for C

    # Force the entries in the diagonal of C to be zero
    reg_lossD = tf.trace(tf.square(C))/n_sample

    cost = loss + lambda1 * reg_lossC + lambda2 * reg_lossD
    optimizer = optimize(cost, learning_rate, optimizer)

    saver = tf.train.Saver()
    # Optimizing the function
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print("Calculating C ...")
        for i in range(1, epochs+1):
            sess.run(optimizer, feed_dict={X: inputX})
            loss = sess.run(cost, feed_dict={X: inputX})
            if i % print_step == 0:
                print('epoch {0}: global loss = {1}'.format(i, loss))
            # if i % 50 == 0:
            #     save_path = saver.save(sess, "./model_C_"+str(i)+".ckpt")
            #     print("Model saved in file: %s" % save_path)

        C_val = sess.run(C)

        return C_val
        # Add ops to save and restore all the variables.


  # Save the variables to disk.
  

        
