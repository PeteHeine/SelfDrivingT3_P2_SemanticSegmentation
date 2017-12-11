import os.path
import glob
import time 
import tensorflow as tf
import helper
import warnings
import numpy as np 
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    print("Loading Model...")
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    print("     Loading Model...")
    graph = tf.get_default_graph()
    l_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    l_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    
    return l_input, l_prob, l_3, l_4, l_7

print("Test1:")
tests.test_load_vgg(load_vgg, tf)
print("     Test1: Done")

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # NOTE:many implementation of conv2d
    # tf.nn.conv2d 
    # tf.layers.conv2d
    # !!! tf.contrib.layers.conv2d  !!!
    # !!! tf.contrib.slim.conv2d   !!!
    # The smart thing about slim and contrib.layer is that initialization and an relu-actication is used automatically. 
    
    # Skip connections... A 1x1 convolutions is used to transform some number
    # of channels into the correct number of classes (num_classes). This is important
    # if future layers are added. (Alternatively they could be concatenated). 
    skip3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
                                                                                    
    
    skip4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    # The final layer of VGG is used at a coarse scale. 
    x = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                         kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    # This is upsampled
    x = tf.layers.conv2d_transpose(x,num_classes,4,2,padding='same',
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    # The upscaled version is now compatible with skip-connection 3 (skip3)
    x = tf.add(x,skip4)
    
    # This is upsampled
    x = tf.layers.conv2d_transpose(x,num_classes,4,2,padding='same',
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    # The upscaled version is now compatible with skip-connection 4 (skip4)
    x = tf.add(x,skip3)
    
    # Finally the output should match the dimensions of the input data. 
    # No activation function is used in the final layer. 
    x = tf.layers.conv2d_transpose(x,num_classes,16,8,padding='same',
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    return x

def layers_2(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    # NOTE:many implementation of conv2d
    # tf.nn.conv2d 
    # tf.layers.conv2d
    # !!! tf.contrib.layers.conv2d  !!!
    # !!! tf.contrib.slim.conv2d   !!!
    # The smart thing about slim and contrib.layer is that initialization and an relu-actication is used automatically. 
    
    # Skip connections... A 1x1 convolutions is used to transform some number
    # of channels into the correct number of classes (num_classes). This is important
    # if future layers are added. (Alternatively they could be concatenated). 
    skip3 = tf.contrib.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                     activation_fn='None', 
                                     weights_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    skip4 = tf.contrib.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                     activation_fn='None', 
                                     weights_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # The final layer of VGG is used at a coarse scale. 
    x = tf.contrib.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                 activation_fn='None', 
                                 weights_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # This is upsampled
    x = tf.contrib.layers.conv2d_transpose(x,num_classes,4,2,padding='same',
                                           activation_fn='None', 
                                           weights_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # The upscaled version is now compatible with skip-connection 3 (skip3)
    x = tf.add(x,skip3)
    
    # This is upsampled
    x = tf.contrib.layers.conv2d_transpose(x,num_classes,4,2,padding='same',
                                           activation_fn='None', 
                                           weights_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # The upscaled version is now compatible with skip-connection 4 (skip4)
    x = tf.add(x,skip4)
    
    # Finally the output should match the dimensions of the input data. 
    # No activation function is used in the final layer. 
    x = tf.contrib.layers.conv2d_transpose(x,num_classes,16,8,padding='same',
                                           activation_fn='None', weights_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    return x
print("Test2:")
tests.test_layers(layers)
print("     Test2: Done")

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    
    combined_loss = cross_entropy_loss+sum(reg_losses)
    
    # Add the optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(combined_loss)
    
    return logits, train_op, cross_entropy_loss

print("Test3:")
tests.test_optimize(optimize)
print("     Test3: Done")

def train_nn(sess, epochs, batch_size, get_batches_fn, logits,train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,model_out):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    print("train_nn started")
    sess.run(tf.global_variables_initializer())
    set_keep_prob = 0.75
    set_learning_rate = 0.0001
    n_samples = 289.0
    print("Starting training")
    plt.figure()
    t_total = time.time()
    for idx in range(epochs):
        print("Epoch {}".format(idx+1))
        epoch_loss = 0
        t1 = time.time()
        for idx_batch,(image,label) in enumerate(get_batches_fn(batch_size)):

            # 50 % change of flipping the image.
            idx_mirror_image = np.random.randint(2,size=(image.shape[0]),dtype=bool)
            for iMirror,do_mirroring in enumerate(idx_mirror_image):
                if do_mirroring:
                    image[iMirror,:,:,:] = np.fliplr(image[iMirror,:,:,:])
                    label[iMirror,:,:] = np.fliplr(label[iMirror,:,:])
            
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: set_keep_prob, learning_rate: set_learning_rate})
            epoch_loss += loss
            print("Epoch ({}/{}), batch({}/{}) Loss: = {:.3f}".format(idx+1,epochs,idx_batch,(n_samples/batch_size),loss))
        
        tnow = time.time()
        print("Epoch {} loss: {}".format(idx+1,epoch_loss))
        print("     Time Total: {}. Epoch: {}".format(tnow-t_total,tnow-t1))
        print()
        
#        # Small test
#        (test) = sess.run([logits], feed_dict={input_image: np.expand_dims(image[0,:,:,:],axis=0), keep_prob: set_keep_prob})
#        plt.imshow(np.reshape(test[0][:,0],(160,576)))            
#        plt.draw()
        
        tf.train.Saver().save(sess, model_out)
            
    print("     Training done.")        
            
    
print("Test4") # Why is the code block by this section? 
#tests.test_train_nn(train_nn)
print("     Test4: Done")


#def run():
print("Run started.")        
num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'
model_out = './models'
tests.test_for_kitti_dataset(data_dir)

# Download pretrained vgg model
helper.maybe_download_pretrained_vgg(data_dir)

# OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
# You'll need a GPU with at least 10 teraFLOPS to train on.
#  https://www.cityscapes-dataset.com/

# Define placeholders
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')

# No GPU was available to me, so it was very time consuming to test various network hyper-parameters. 
# Learning rate, batch_size, epochs and the amount of regulization was found in submissions from
# other udacity submission.

batch_size = 8
epochs = 20

print("Starting Session")
with tf.Session() as sess:
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    
    print("Loading model")
    
    n_tsamples = len(glob.glob(os.path.join(data_dir, 'data_road/training/image_2/*')))
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    
    
    print("     Loading model done...")
    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # TODO: Build NN using load_vgg, layers, and optimize function
    l_input, keep_prob, l_3, l_4, l_7 = load_vgg(sess,vgg_path)
    net = layers(l_3, l_4, l_7,num_classes)
    logits, train_op, cross_entropy_loss = optimize(net,correct_label,learning_rate,num_classes)
    
    
#    sess.run(tf.global_variables_initializer())
#    set_keep_prob = 0.75
#    set_learning_rate = 0.0001
#    image,label = next(get_batches_fn(8))
#    (test) = sess.run([logits], feed_dict={l_input: image[:2,:,:,:], keep_prob: 0.75})
#    print("test: ", test[0].shape)
#    plt.figure()
#    plt.imshow(np.reshape(test[0][:,0],(160,576,2))[:,:,1])
    
    # TODO: Train NN using the train_nn function
    train_nn(sess,epochs,batch_size,get_batches_fn,logits,train_op,cross_entropy_loss,l_input,correct_label,keep_prob,learning_rate,model_out)

    # TODO: Save inference data using helper.save_inference_samples
    #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

    # Save model
    tf.train.Saver().save(sess, model_out)

    # TODO: Save inference data using helper.save_inference_samples
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, l_input)
    
    # OPTIONAL: Apply the trained model to a video





#np.random.randint(1,)
#if __name__ == '__main__':
#    run()
