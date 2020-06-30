```
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
    
mnist_train_x=None
mnist_train_y=None
mnist_test_x=None
mnist_test_y=None

class Linear_Regression:
    learning_rate=1e-2
    global_step=tf.Variable(0, trainable=False, name='global_step')

    X_data=None
    Y_data=None
    _X=tf.placeholder(tf.float32)
    _Y=tf.placeholder(tf.float32)
    bias=None
    weight=None

    hypothesis=None

    sess = tf.Session()

    loss_function=None
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    _train_op=None

    saver=None
    ckpt_name="linear_regression"
    restore=None

    def __init__(self, restore=False, ckpt_name=None):
        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self.restore=restore
        self.weight=tf.Variable(tf.random_uniform([1], -1., 1.))
        self.bias=tf.Variable(tf.random_uniform([1], -1., 1.))
        self.hypothesis=self.weight*self._X+self.bias
        self.loss_function=tf.reduce_mean(tf.square(self.hypothesis - self._Y))
        self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)

        self.saver=tf.train.Saver(tf.global_variables())

        ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
        if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            _init = tf.global_variables_initializer()
            self.sess.run(_init)

    def train(self, times=100, print_every=10):
        dtime=time.time()
        
        if self.Y_data is not None :
            for step in range(times):
                self.sess.run(self._train_op, feed_dict={self._X:self.X_data, self._Y:self.Y_data})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self._X: self.X_data, self._Y: self.Y_data}))
        else :
            for step in range(times):
                self.sess.run(self._train_op, feed_dict={self._X:self.X_data})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self._X: self.X_data}))

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.saver.save(self.sess, "./"+self.ckpt_name+"_models/model.ckpt", global_step=self.global_step)

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        if self.Y_data is not None :
            return self.sess.run(self.hypothesis,feed_dict={self._X:inputs, self._Y:self.Y_data})
        else:
            return self.sess.run(self.hypothesis,feed_dict={self._X:inputs})

class Logistic_Regression(Linear_Regression):
    ckpt_name="logistic_regression"

    def __init__(self, input_size=1, restore=False, ckpt_name=None):
        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self.restore=restore
        self.weight=tf.Variable(tf.random_normal([input_size, 1], mean=0.01, stddev=0.01))
        self.bias=tf.Variable(tf.random_normal([1]))
        self.hypothesis=tf.nn.sigmoid(tf.matmul(self._X,self.weight)+self.bias)
        self.loss_function=-tf.reduce_mean(self._Y*tf.log(self.hypothesis)+(1-self._Y)*(tf.log(1-self.hypothesis)))
        self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)

        self.saver=tf.train.Saver(tf.global_variables())

        ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
        if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            _init = tf.global_variables_initializer()
            self.sess.run(_init)

class Perceptron:
    learning_rate=1e-2
    global_step=tf.Variable(0, trainable=False, name='global_step')

    _opsize=None
    softmax=True

    X_data=None
    Y_data=None
    _X=tf.placeholder(tf.float32)
    _Y=tf.placeholder(tf.float32)
    bias=None
    weight=None
    
    model=None
    _mnist_loaded=False
    
    sess = tf.Session()

    loss_function=None
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    _train_op=None

    saver=None
    ckpt_name="perceptron"
    restore=None

    def __init__(self, input_size, output_size=1, restore=False, ckpt_name=None, softmax=True):
        self._opsize=output_size

        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self.restore=restore
        self.softmax=softmax
        self.weight=tf.Variable(tf.random_normal([input_size, output_size], mean=0.01, stddev=0.01))
        self.bias=tf.Variable(tf.random_normal([output_size]))
        self.model=tf.nn.sigmoid(tf.add(tf.matmul(self._X, self.weight),self.bias))
        if self.softmax and self._opsize > 1 :
            self.loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self._Y))
        else:
            self.loss_function=tf.reduce_mean(tf.reduce_sum(tf.square(self._Y - self.model), axis=1))
        self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)

        self.saver=tf.train.Saver(tf.global_variables())

        ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
        if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            _init = tf.global_variables_initializer()
            self.sess.run(_init)

    def train(self, times=100, print_every=10):
        dtime=time.time()
        
        if self.Y_data is not None :
            for step in range(times):
                self.sess.run(self._train_op, feed_dict={self._X:self.X_data, self._Y:self.Y_data})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self._X: self.X_data, self._Y: self.Y_data}))
        elif self.Y_data is None and self.X_data is None :
            for step in range(times):
                self.sess.run(self._train_op)

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function))
        else :
            for step in range(times):
                self.sess.run(self._train_op, feed_dict={self._X:self.X_data})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self._X: self.X_data}))

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")

        self.saver.save(self.sess, "./"+self.ckpt_name+"_models/model.ckpt", global_step=self.global_step)

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        model=self.model
        if self.softmax and self._opsize > 1 :
                model=tf.nn.softmax(model)

        if self.Y_data is not None :
            return self.sess.run(model,feed_dict={self._X:inputs, self._Y:self.Y_data})
        else:
            return self.sess.run(model,feed_dict={self._X:inputs})

class ANN(Perceptron):
    layer=None
    ckpt_name="ANN"

    def __init__(self, input_size, hidden_size=10, output_size=1, restore=False, ckpt_name=None, softmax=True):
        self._opsize=output_size
        self.softmax=softmax

        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self.restore=restore
        self.weight=[]
        self.weight.append(tf.Variable(tf.random_normal([input_size, hidden_size], mean=0.01, stddev=0.01)))
        self.layer=tf.nn.relu(tf.matmul(self._X, self.weight[0]))
        self.weight.append(tf.Variable(tf.random_normal([hidden_size, output_size], mean=0.01, stddev=0.01)))
        self.bias=tf.Variable(tf.random_normal([output_size]))
        self.model=tf.add(tf.matmul(self.layer, self.weight[1]),self.bias)
        if self.softmax and self._opsize > 1 :
            self.loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self._Y))
        else:
            self.loss_function=tf.reduce_mean(tf.reduce_sum(tf.square(self._Y - self.model), axis=1))
        self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)
        self.saver=tf.train.Saver(tf.global_variables())

        ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
        if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            _init = tf.global_variables_initializer()
            self.sess.run(_init)

class DNN(ANN):
    level=0
    _hdsize=10
    _ipsize=None
    _opsize=1
    ckpt_name="DNN"

    def __init__(self, input_size, hidden_size=10, output_size=1, layer_level=3, restore=False, ckpt_name=None, softmax=True):
        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self._ipsize=input_size
        self._hdsize=hidden_size
        self._opsize=output_size
        self.restore=restore
        self.softmax=softmax

        if layer_level < 1:
            print("Please set a layer level at least 1.")
        else:
            self.level=layer_level
            self.weight=[]
            self.layer=[]
            self.weight.append(tf.Variable(tf.random_normal([input_size, hidden_size], mean=0.01, stddev=1)))
            self.layer.append(tf.nn.relu(tf.matmul(self._X, self.weight[0])))
            for i in range(1,layer_level):
                self.weight.append(tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.01, stddev=1)))
                self.layer.append(tf.nn.relu(tf.matmul(self.layer[i-1], self.weight[i])))
            self.weight.append(tf.Variable(tf.random_normal([hidden_size, output_size], mean=0.01, stddev=1)))
            self.bias=tf.Variable(tf.random_normal([output_size]))
            
            self.model=tf.add(tf.matmul(self.layer[-1], self.weight[-1]),self.bias)
            if self.softmax and self._opsize > 1 :
                self.loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self._Y))
            else:
                self.loss_function=tf.reduce_mean(tf.reduce_sum(tf.square(self._Y - self.model), axis=1))
            self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)
            self.saver=tf.train.Saver(tf.global_variables())

            ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                _init = tf.global_variables_initializer()
                self.sess.run(_init)

class CNN(DNN):
    ckpt_name="CNN"
    conv_layers=[]
    conv_level=2
    _kernel_size=32
    _input_level=1
    mnist_train_x=None
    mnist_train_y=None
    mnist_test_x=None
    mnist_test_y=None
    _X=None
    _Y=None
    is_train=tf.placeholder(tf.bool)
    _pre_batch_pos=0
    gray=True

    def __init__(self, input_size=[28,28], input_level=1, kernel_size=[3,3], kernel_count=32, strides=[1,1], hidden_size=128, output_size=1, conv_level=2, layer_level=1, restore=False, ckpt_name=None, softmax=True):
        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self._input_level=input_level
        self._ipsize=input_size
        self._hdsize=hidden_size
        self._opsize=output_size
        self.restore=restore
        self._kernel_size=kernel_size
        self._X=tf.placeholder(tf.float32, [None,input_size[0],input_size[1],input_level])
        self._Y=tf.placeholder(tf.float32, [None,output_size])

        if input_level == 3 :
            self.gray=False
        
        if layer_level < 1 :
            print("Please set a Fully-connected layer level at least 1 as 'set_layer_level(n)'.")
        elif conv_level < 1 :
            print("Please set a Convolutional layer level at least 1 as 'set_conv_level(n)'.")
        else:
            self.level=layer_level
            self.conv_level=conv_level
            self.weight=[]
            self.layer=[]

            self.conv_layers.append(tf.layers.conv2d(self._X, kernel_count, kernel_size, strides=strides, activation=tf.nn.relu, padding='SAME'))
            self.conv_layers[0]=tf.layers.max_pooling2d(self.conv_layers[0], [2,2], [2,2], padding='SAME')
            self.conv_layers[0]=tf.layers.dropout(self.conv_layers[0], 0.7, self.is_train)
            for i in range(1,conv_level):
                self.conv_layers.append(tf.layers.conv2d(self.conv_layers[i-1], kernel_count, kernel_size, strides=strides, padding='SAME'))
                self.conv_layers[i]=tf.layers.max_pooling2d(self.conv_layers[i], [2,2], [2,2], padding='SAME')
                self.conv_layers[i]=tf.layers.dropout(self.conv_layers[i], 0.7, self.is_train)

            self.layer.append(tf.layers.flatten(self.conv_layers[-1]))
            self.layer[0]=tf.layers.dense(self.layer[0], hidden_size, activation=tf.nn.relu)
            self.layer[0]=tf.layers.dropout(self.layer[0], 0.5, self.is_train)
            for i in range(1,layer_level):
                self.layer.append(tf.layers.dense(self.layer[i-1], hidden_size, activation=tf.nn.relu))
                self.layer[i]=tf.layers.dropout(self.layer[i], 0.5, self.is_train)

            self.model=tf.layers.dense(self.layer[-1], output_size, activation=None)
            if self.softmax and self._opsize > 1 :
                self.loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self._Y))
            else:
                self.loss_function=tf.reduce_mean(tf.reduce_sum(tf.square(self._Y - self.model), axis=1))
            self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)
            self.saver=tf.train.Saver(tf.global_variables())

            ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                _init = tf.global_variables_initializer()
                self.sess.run(_init)

    def train(self, times=100, batch=500, print_every=10):
        dtime=time.time()

        if self.Y_data is not None :
            for step in range(times):
                pos=self._pre_batch_pos
                self.sess.run(self._train_op, feed_dict={self._X:self.X_data[pos:pos+batch], self._Y:self.Y_data[pos:pos+batch], self.is_train:True})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self._X: self.X_data[pos:pos+batch], self._Y: self.Y_data[pos:pos+batch], self.is_train:False}))
                
                self._pre_batch_pos+=batch
                self._pre_batch_pos%=len(self.X_data)
        elif self.Y_data is None and self.X_data is None :
            for step in range(times):
                self.sess.run(self._train_op, feed_dict={self.is_train:True})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self.is_train:False}))
        else :
            for step in range(times):
                pos=self._pre_batch_pos
                self.sess.run(self._train_op, feed_dict={self._X:self.X_data[pos:pos+batch], self.is_train:True})

                if (step + 1) % print_every == 0:
                    print(self.sess.run(self.global_step), "step loss:", self.sess.run(self.loss_function, feed_dict={self._X: self.X_data[pos:pos+batch], self.is_train:False}))
                
                self._pre_batch_pos+=batch
                self._pre_batch_pos%=len(self.X_data)

        print("Training is done.\nTime spent:", round(time.time()-dtime,1), "s\nTraining speed:", round(times/(time.time()-dtime),1), "step/s")
        self._pre_batch_pos=batch

        self.saver.save(self.sess, "./"+self.ckpt_name+"_models/model.ckpt", global_step=self.global_step)

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        if len(inputs)<5:
            for input in inputs:
                if self.gray:
                    plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]), cmap='gray', vmin=0, vmax=255)
                else:
                    plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]))
                plt.show()

        model=self.model
        if self.softmax and self._opsize > 1 :
                model=tf.nn.softmax(model)

        if self.Y_data is not None :
            return self.sess.run(model,feed_dict={self._X:inputs, self._Y:self.Y_data})
        else:
            return self.sess.run(model,feed_dict={self._X:inputs})

    def load_MNIST(self):
        global mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y
        if mnist_train_x is None and mnist_train_y is None:
            (self.mnist_train_x,self.mnist_train_y), (self.mnist_test_x, self.mnist_test_y) = tf.keras.datasets.mnist.load_data()
            self.mnist_train_x=self.mnist_train_x.reshape(-1, 28, 28, 1)
            self.mnist_train_y=tf.one_hot(self.mnist_train_y, 10)
            self.mnist_test_x=self.mnist_test_x.reshape(-1, 28, 28, 1)
            self.mnist_test_y=tf.one_hot(self.mnist_test_y, 10)
            self.X_data=self.mnist_train_x
            self.Y_data=self.sess.run(self.mnist_train_y)
        else:
            self.mnist_train_x=mnist_train_x
            self.mnist_train_y=mnist_train_y
            self.mnist_test_x=mnist_test_x
            self.mnist_test_y=mnist_test_y
            self.mnist_train_x=self.mnist_train_x.reshape(-1, 28, 28, 1)
            self.mnist_train_y=tf.one_hot(self.mnist_train_y, 10)
            self.mnist_test_x=self.mnist_test_x.reshape(-1, 28, 28, 1)
            self.mnist_test_y=tf.one_hot(self.mnist_test_y, 10)
            self.X_data=self.mnist_train_x
            self.Y_data=self.sess.run(self.mnist_train_y)

    def show_img(self, input):
        if self.gray:
            plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]), cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(input.reshape(self._ipsize[0],self._ipsize[1]))
        plt.show()


class DQN(DNN):
    level=0
    _hdsize=10
    _stsize=None
    _opsize=1
    ckpt_name="DQN"
    actions=tf.placeholder(tf.float32)
    rewards=tf.placeholder(tf.float32)
    prob=None
    def __init__(self, state_size, hidden_size=5, output_size=1, layer_level=1, restore=False, ckpt_name=None):
        if ckpt_name is not None:
            self.ckpt_name=ckpt_name
        self._stsize=state_size
        self._hdsize=hidden_size
        self._opsize=output_size
        self.restore=restore

        if layer_level < 1:
            print("Please set a layer level at least 1.")
        else:
            self.level=layer_level
            self.weight=[]
            self.layer=[]
            self.weight.append(tf.Variable(tf.random_normal([state_size, hidden_size], mean=0.01, stddev=0.01)))
            self.layer.append(tf.matmul(self._X, self.weight[0]))
            for i in range(1,layer_level):
                self.weight.append(tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.01, stddev=0.01)))
                self.layer.append(tf.matmul(self.layer[i-1], self.weight[i]))
            self.weight.append(tf.Variable(tf.random_normal([hidden_size, output_size], mean=0.01, stddev=0.01)))
            self.bias=tf.Variable(tf.random_normal([output_size]))
            self.prob=tf.reduce_mean(tf.reduce_mean(tf.add(tf.matmul(self.layer[-1], self.weight[-1]),self.bias)))
            self.model= np.bool(self.prob>=0.5)
            self.loss_function=tf.pow(1/tf.sqrt(tf.reduce_mean(tf.abs(self.actions - self.prob) * self.rewards))*10,2.)
            self._train_op=self.optimizer.minimize(self.loss_function, global_step=self.global_step)
            self.saver=tf.train.Saver(tf.global_variables())

            ckpt=tf.train.get_checkpoint_state("./"+self.ckpt_name+"_models")
            if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)) and restore:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                _init = tf.global_variables_initializer()
                self.sess.run(_init)


    def train(self, states, rewards, actions, reward_std="time"):
        if reward_std=="time":
            rewards=self.process_rewards(rewards)
            self.sess.run(self._train_op, feed_dict={self._X: states, self.rewards: rewards, self.actions: actions})

        loss=self.sess.run(self.loss_function, feed_dict={self._X: states, self.rewards: rewards, self.actions: actions})

        return loss

    def run(self,inputs=None):
        if inputs is None:
            inputs=self.X_data
            print("run with : ",inputs)

        return self.sess.run(self.model,feed_dict={self._X:inputs})

    def save(self):
        self.saver.save(self.sess, "./"+self.ckpt_name+"_models/model.ckpt", global_step=self.global_step)

    def process_rewards(self, r):
        dr = np.zeros_like(r)

        dr[-1] = r[-1]
        for t in reversed(range(0, len(r)-1)):
            dr[t] = dr[t+1] + r[t]
        
        return dr


def onehot(array, classes):
    arr=np.array(array)
    return np.squeeze(np.eye(classes)[arr.reshape(-1)])
```
