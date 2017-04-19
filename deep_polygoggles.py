
# coding: utf-8

# # Porting Tensorflow tutorial "Deep MNIST for Experts" to polygoggles
#
# based on https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html

# In[21]:

import math
import tensorflow as tf

import datasets
import make_polygon_pngs



def get_data_sets_and_params(use_MNIST_instead_of_our_data=False):
    if use_MNIST_instead_of_our_data:
        params = dict(
            width = 28,
            height = 28,
            num_training_steps = 20000,
            batch_size = 50,
        )
    else:
        params = dict(
            width = 70,
            height = 70,
            num_training_steps = 1000,
            batch_size = 50,
            training_images = 5000,
            test_images = 1000,
            allow_rotation = True,
        )

    if use_MNIST_instead_of_our_data:
        from tensorflow.examples.tutorials.mnist import input_data
        data_sets = input_data.read_data_sets('MNIST_data', one_hot=True)
    else:
        collection_dir = make_polygon_pngs.make_collection(params['width'],
                                                           params['height'],
                                                           params['training_images'],
                                                           params['test_images'],
                                                           allow_rotation=params['allow_rotation'])
        data_sets = datasets.read_data_sets(collection_dir)
    return data_sets, params



def run_regression(data_sets):
    sess = tf.InteractiveSession()

    flat_size = width * height
    num_labels = data_sets.train.labels.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, flat_size])
    y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
    W = tf.Variable(tf.zeros([flat_size, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))

    sess.run(tf.initialize_all_variables())

    # We can now implement our regression model. It only takes one line!
    # We multiply the vectorized input images x by the weight matrix W, add the bias b,
    # and compute the softmax probabilities that are assigned to each class.
    y = tf.nn.softmax(tf.matmul(x,W) + b)

    # The cost function to be minimized during training can be specified just as easily.
    # Our cost function will be the cross-entropy between the target and the model's prediction.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    # Now that we have defined our model and training cost function, it is straightforward to train using TensorFlow.
    # Because TensorFlow knows the entire computation graph, it can use automatic differentiation to find
    # the gradients of the cost with respect to each of the variables.
    # TensorFlow has a variety of builtin optimization algorithms.
    # For this example, we will use steepest gradient descent, with a step length of 0.01, to descend the cross entropy.
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # What TensorFlow actually did in that single line was to add new operations to the computation graph.
    # These operations included ones to compute gradients, compute parameter update steps, and apply update
    # steps to the parameters.
    #
    # The returned operation train_step, when run, will apply the gradient descent updates to the parameters.
    # Training the model can therefore be accomplished by repeatedly running train_step.

    for i in range(1000):
        batch = data_sets.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    # That gives us a list of booleans. To determine what fraction are correct, we cast to floating point
    # numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Finally, we can evaluate our accuracy on the test data. (On MNIST this should be about 91% correct.)

    accuracy = accuracy.eval(feed_dict={x: data_sets.test.images, y_: data_sets.test.labels})
    print("Accuracy: %.5f" % accuracy)
    return accuracy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def run_multilayer_convolutional_networkd():
    """
    # # Build a Multilayer Convolutional Network
    #
    # Getting .91 accuracy on MNIST is bad. It's almost embarrassingly bad. In this section, we'll fix that, jumping from a very simple model to something moderately sophisticated: a small convolutional neural network. This will get us to around 99.2% accuracy -- not state of the art, but respectable.
    #
    # ## Weight Initialization
    #
    # To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons." Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.
    """
    # ## Convolution and Pooling
    #
    # TensorFlow also gives us a lot of flexibility in convolution and pooling operations.
    # How do we handle the boundaries? What is our stride size? In this example, we're always going
    # to choose the vanilla version. Our convolutions uses a stride of one and are zero padded so
    # that the output is the same size as the input.
    # Our pooling is plain old max pooling over 2x2 blocks. To keep our code cleaner,
    # let's also abstract those operations into functions.



    # ## First Convolutional Layer
    #
    # We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.

    # In[16]:

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])


    # To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.

    # In[17]:

    x_image = tf.reshape(x, [-1, width, height,1]) # XXX not sure which is width and which is height


    # In[18]:

    # We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    # ## Second Convolutional Layer
    #
    # In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.

    # In[19]:

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # ## Densely Connected Layer
    #
    # Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.

    # XXX where is the 7x7 coming from?
    #
    # when bumping to width, height of 50 each:
    #
    # InvalidArgumentError: Input to reshape is a tensor with 540800 values, but the requested shape requires a multiple of 3136
    #
    # 7 x 7 x 64 = 3136
    #
    #
    # 540800 / 64. = 8450
    #
    # 13 x 13 x 50 x 64 = 540800
    #
    #
    #
    #
    #
    # On MNIST, if I change the densely connected layer to fail (change the 7x7x64 to 7x7x65 in both W_fcl and h_pool2_flat
    # for example, then I get the following error as soon as start to train:
    #
    # InvalidArgumentError: Input to reshape is a tensor with 156800 values, but the requested shape requires a multiple of 3185
    #
    # note 3185 = 7x7x65
    #
    # 156800 = 7 * 7 * 64 * 50
    #
    # 50 is batch size
    #
    # ##### with width & height = 70:
    # Input to reshape is a tensor with 1036800 values, but the requested shape requires a multiple of 10816
    #
    # ##### with width & height = 150:
    # Input to reshape is a tensor with 4620800 values, but the requested shape requires a multiple of 20736

    # In[23]:

    def get_size_reduced_to_from_input_tensor_size(input_tensor_size):
        size_reduced_to_squared = input_tensor_size / 64. / batch_size # last divide is 50., pretty sure it's batch size
        return math.sqrt(size_reduced_to_squared)
    print(get_size_reduced_to_from_input_tensor_size(4620800))
    print(get_size_reduced_to_from_input_tensor_size(1036800))


    # In[24]:

    if use_MNIST_instead_of_our_data:
        size_reduced_to = 7
    else:
        # for width & height = 50, size_reduced_to seems to be 13
        # for width & height = 70, size_reduced_to seems to be 18
        # for width & height = 150, size_reduced_to seems to be 38
        size_reduced_to = 18

    #W_fc1 = weight_variable([7 * 7 * 64, 1024])
    W_fc1 = weight_variable([size_reduced_to * size_reduced_to * 64, 1024])
    b_fc1 = bias_variable([1024])

    #h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_pool2_flat = tf.reshape(h_pool2, [-1, size_reduced_to*size_reduced_to*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # #### Dropout
    #
    # To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.

    # In[25]:

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # ## Readout Layer
    #
    # Finally, we add a softmax layer, just like for the one layer softmax regression above.

    # In[26]:

    W_fc2 = weight_variable([1024, num_labels])
    b_fc2 = bias_variable([num_labels])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # ## Train and Evaluate the Model
    #
    # How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above. The differences are that: we will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer; we will include the additional parameter keep_prob in feed_dict to control the dropout rate; and we will add logging to every 100th iteration in the training process.

    # In[27]:

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())


    # In[ ]:

    for i in range(num_training_steps):
        batch = data_sets.train.next_batch(batch_size)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


# In[ ]:
