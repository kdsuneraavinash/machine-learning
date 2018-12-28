# Machine Learning

## Tensorflow

### Installing Tensorflow-gpu

1. Install GPU Drivers. (No need to install CuDNN which will be installed by Anaconda)

2. Install [Anaconda](https://www.anaconda.com/downloads). Do not add to PATH at the installation. Confirm by typing `py` and `py -2`. These will give locally installed python 2 and 3. Then add `Anaconda3/` and `Anaconda3/scripts` to the PATH at the top. Now `py` should give local python and python should give Anaconda installation.

3. Create a new environment.

4. Install `tensorflow-gpu`.

5. [Test Script](https://www.tensorflow.org/guide/using_gpu)

   ```python
   import tensorflow as tf
   
   # Creates a graph.
   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
   c = tf.matmul(a, b)
   # Creates a session with log_device_placement set to True.
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
   # Runs the op.
   print(sess.run(c))
   
   '''
   =================================================================
   EXPECTED OUTPUT
   ==================================================================
   
   Device mapping:
   /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
   id: 0000:05:00.0
   b: /job:localhost/replica:0/task:0/device:GPU:0
   a: /job:localhost/replica:0/task:0/device:GPU:0
   MatMul: /job:localhost/replica:0/task:0/device:GPU:0
   [[ 22.  28.]
    [ 49.  64.]]
   '''
   
   '''
   =================================================================
   MY OUTPUT
   ==================================================================
   
   2018-12-28 17:12:31.062263: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
   2018-12-28 17:12:31.264979: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
   name: GeForce GTX 1050 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.392
   pciBusID: 0000:01:00.0
   totalMemory: 4.00GiB freeMemory: 3.30GiB
   2018-12-28 17:12:31.269083: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
   2018-12-28 17:12:31.653322: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
   2018-12-28 17:12:31.655784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
   2018-12-28 17:12:31.656878: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
   2018-12-28 17:12:31.658032: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3011 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
   Device mapping:
   /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
   2018-12-28 17:12:31.663121: I tensorflow/core/common_runtime/direct_session.cc:307] Device mapping:
   /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
   
   MatMul: (MatMul): /job:localhost/replica:0/task:0/device:GPU:0
   2018-12-28 18:03:38.555255: I tensorflow/core/common_runtime/placer.cc:927] MatMul: (MatMul)/job:localhost/replica:0/task:0/device:GPU:0
   a: (Const): /job:localhost/replica:0/task:0/device:GPU:0
   2018-12-28 18:03:38.558586: I tensorflow/core/common_runtime/placer.cc:927] a: (Const)/job:localhost/replica:0/task:0/device:GPU:0
   b: (Const): /job:localhost/replica:0/task:0/device:GPU:0
   2018-12-28 18:03:38.560624: I tensorflow/core/common_runtime/placer.cc:927] b: (Const)/job:localhost/replica:0/task:0/device:GPU:0
   [[22. 28.]
    [49. 64.]]
   '''
   ```

6. Can change environment through  Anaconda Prompt using `activate ENVNAME`.

7. Warning `Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2` was ignored. ([Stack Overflow](https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u)) Initially tried to fixed by ended up corrupting the installation. Tried method is given below. DO NOT DO IT.

   1. Find correct `.whl` link using [this link](https://github.com/lakshayg/tensorflow-build). (In my case [this](https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.9.0/py36/GPU/cuda92cudnn71avx2/tensorflow_gpu-1.9.0-cp36-cp36m-win_amd64.whl) for windows, python 3.6 and CuDNN 7.1 (Version can be found in `cudnn.h` file in Anaconda installation directory))

   2. Install using `pip install --ignore-installed --upgrade "Download URL"`. (Need to have closed all anaconda programs except the Anaconda Prompt.)

   3. Got the error `tensorflow 1.12.0 has requirement tensorboard<1.13.0,>=1.12.0, but you'll have tensorboard 1.9.0 which is incompatible.`

      1. Ran following commands.

         ```
         pip uninstall tensorflow-tensorboard
         pip uninstall protobuf
         pip uninstall tensorboard
         pip install tensorflow-gpu
         pip install --upgrade tensorflow
         ```

      2. Still got the error. And now modules couldn't be imported in Python. **So restarted the procedure in a new environment.**

### Confirming if gpu is used

Use GPU-Z to monitor GPU Load. This should increased when training. Use following model. (Need to have a empty `mnist` directory beside the script file)

```python
import sys, os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_pah = os.path.join(sys.path[0], "mnist")
mnist = input_data.read_data_sets(mnist_pah, one_hot = True)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
```

## LOG

> 2018/12/28: Installed tensorflow-gpu
>
> 2018/12/29: Trained a model on MNIST dataset