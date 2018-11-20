import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model,model_from_json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD,RMSprop,Adagrad,Adadelta

import os
import random
import time
import wget
import tarfile
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_train_history_acc(his,acc1,val_acc1):
    plt.plot(his.history[acc1],'r')
    plt.plot(his.history[val_acc1],'g')
    plt.title('ACC History')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['TACC','VACC'],loc = 'upper left')
    plt.savefig('train-val-ACC.png')
def show_train_history_loss(his,loss1,val_loss1):
    plt.plot(his.history[loss1],'b')
    plt.plot(his.history[val_loss1],'y')
    plt.title('LOSS History')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['TLOSS','VLOSS'],loc = 'upper left')
    plt.savefig('train-val-LOSS.png')
class MaxUnpoolWithArgmax(Layer):

    def __init__(self, pooling_argmax, stride = [1, 2, 2, 1], **kwargs):
        self.pooling_argmax = pooling_argmax    
        self.stride = stride
        super(MaxUnpoolWithArgmax, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MaxUnpoolWithArgmax, self).build(input_shape)

    def call(self, inputs):
        #input_shape = K.cast(K.shape(inputs), dtype='int64')
        #input_shape=K.cast(inputs.shape,dtype='int64')
        input_shape=inputs.shape
        output_shape = (input_shape[0],
                        input_shape[1] * self.stride[1],
                        input_shape[2] * self.stride[2],
                        input_shape[3])
        #output_list = []
        #output_list.append(self.pooling_argmax // (output_shape[2] * output_shape[3]))
        #output_list.append(self.pooling_argmax % (output_shape[2] * output_shape[3]) // output_shape[3])
        argmax = self.pooling_argmax #K.stack(output_list)
        
        one_like_mask = K.ones_like(argmax)
        
        batch_range = K.reshape(K.arange(start=0, stop=input_shape[0], dtype='int64'), 
                                 shape=[input_shape[0], 1, 1, 1])

        b = one_like_mask * batch_range
        y = argmax // (output_shape[2] * output_shape[3])
        x = argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = K.arange(start=0, stop=output_shape[3], dtype='int64')
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(inputs)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(inputs, [updates_size])
        return tf.scatter_nd(indices, values, output_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])

    def get_config(self):
        base_config = super(MaxUnpoolWithArgmax, self).get_config()
        base_config['pooling_argmax'] = self.pooling_argmax
        base_config['stride'] = self.stride
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DeconvNet:
    def __init__(self, use_cpu=False, print_summary=False,batch_size=4,labels=4):
        #self.maybe_download_and_extract()
        self.batch_size=batch_size
        self.labels=labels
        self.build(use_cpu=use_cpu, print_summary=print_summary)
        

        
    def maybe_download_and_extract(self):
        """Download and unpack VOC data if data folder only contains the .gitignore file"""
        if os.listdir('data') == ['.gitignore']:
            filenames = ['VOC_OBJECT.tar.gz', 'VOC2012_SEG_AUG.tar.gz', 'stage_1_train_imgset.tar.gz', 'stage_2_train_imgset.tar.gz']
            url = 'http://cvlab.postech.ac.kr/research/deconvnet/data/'

            for filename in filenames:
                wget.download(url + filename, out=os.path.join('data', filename))

                tar = tarfile.open(os.path.join('data', filename))
                tar.extractall(path='data')
                tar.close()

                os.remove(os.path.join('data', filename))

                
    def predict(self, image):
        return self.model.predict(image)
    
    #def save(self, file_path='model.h5'):
        #print(self.model.to_json())
        #self.model.save(file_path)
        
    def load(self, file_path='model.h5'):
        self.model.load_weights(file_path)
        print('='*30+'Model Load Done!!'+'='*30)
        
        
    
    def random_crop_or_pad(self, image, truth, size=(224, 224)):
        assert image.shape[:2] == truth.shape[:2]

        if image.shape[0] > size[0]:
            crop_random_y = random.randint(0, image.shape[0] - size[0])
            image = image[crop_random_y:crop_random_y + size[0],:,:]
            truth = truth[crop_random_y:crop_random_y + size[0],:]
        else:
            zeros = np.zeros((size[0], image.shape[1], image.shape[2]), dtype=np.float32)
            zeros[:image.shape[0], :image.shape[1], :] = image                                          
            image = np.copy(zeros)
            zeros = np.zeros((size[0], truth.shape[1]), dtype=np.float32)
            zeros[:truth.shape[0], :truth.shape[1]] = truth
            truth = np.copy(zeros)

        if image.shape[1] > size[1]:
            crop_random_x = random.randint(0, image.shape[1] - size[1])
            image = image[:,crop_random_x:crop_random_x + 224,:]
            truth = truth[:,crop_random_x:crop_random_x + 224]
        else:
            zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
            zeros[:image.shape[0], :image.shape[1], :] = image
            image = np.copy(zeros)
            zeros = np.zeros((truth.shape[0], size[1]))
            zeros[:truth.shape[0], :truth.shape[1]] = truth
            truth = np.copy(zeros)            

        return image, truth

    #(0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car , 8=cat, 9=chair, 
    # 10=cow, 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 
    # 17=sheep, 18=sofa, 19=train, 20=tv/monitor, 255=no_label)

    def max_pool_with_argmax(self, x):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def BatchGenerator(self, train_stage=1, batch_size=8, image_size=(224, 224, 3), labels=4):
        if train_stage == 1:
            trainset_val = open('data/train.txt').readlines()
        else:
            trainset_val = open('data/train.txt').readlines()

        while True:
            images_val = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
            truths_val = np.zeros((batch_size, image_size[0], image_size[1], labels))

            for i in range(batch_size):
                random_line = random.choice(trainset_val)
                image_file = random_line.split(' ')[0]
                truth_file = random_line.split(' ')[1]
                image_val = np.float32(cv2.imread(image_file)/255.0)

                truth_mask_val = cv2.imread(truth_file[:-1], cv2.IMREAD_GRAYSCALE)
                #truth_mask_val[truth_mask_val == 255] = 10 # replace no_label with background  
                images_val[i], truth_val = self.random_crop_or_pad(image_val, truth_mask_val, image_size)
                truths_val[i] = (np.arange(labels) == truth_val[...,None]).astype(int) # encode to one-hot-vector
            #print('1'*100+'\n',images_val)
            #print('1'*100+'\n',images_val.shape)
            #print('1'*100+'\n',truths_val)
            #print('1'*100+'\n',truths_val.shape)
            
            yield images_val, truths_val
            
    def BatchGenerator_test(self, train_stage=1, batch_size=8, image_size=(224, 224, 3), labels=4):
        if train_stage == 1:
            trainset = open('data/test.txt').readlines()
        else:
            trainset = open('data/test.txt').readlines()

        while True:
            images = np.zeros((batch_size, image_size[0], image_size[1], image_size[2]))
            truths = np.zeros((batch_size, image_size[0], image_size[1], labels))

            for i in range(batch_size):
                random_line = random.choice(trainset)
                image_file = random_line.split(' ')[0]
                truth_file = random_line.split(' ')[1]
                image = np.float32(cv2.imread(image_file)/255.0)

                truth_mask = cv2.imread(truth_file[:-1], cv2.IMREAD_GRAYSCALE)
                truth_mask[truth_mask == 255] = 0 # replace no_label with background  
                images[i], truth = self.random_crop_or_pad(image, truth_mask, image_size)
                truths[i] = (np.arange(labels) == truth[...,None]).astype(int) # encode to one-hot-vector
            yield images, truths

        
    #def train(self, steps_per_epoch=1000, epochs=10, batch_size=32):
    def train(self, steps_per_epoch=1000, epochs=10):
        batch_size=self.batch_size
        print('='*30+'Sart Training'+'='*30)
        batch_generator = self.BatchGenerator(batch_size=batch_size)
        validation_generator=self.BatchGenerator_test(batch_size=batch_size)
        
        history1=self.model.fit_generator(batch_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,validation_data=validation_generator,validation_steps=400)
        print('='*30+'Training Done!!'+'='*30)
        file_path='model.h5'
        self.model.save_weights(file_path)
        print('='*30+'Model Save Done!!'+'='*30)
        show_train_history_acc(history1,'acc','val_acc')
        plt.figure()
        show_train_history_loss(history1,'loss','val_loss')
        

    

    def buildConv2DBlock(self, block_input, filters, block, depth):
        for i in range(1, depth + 1):
            if i == 1:
                conv2d = Conv2D(filters, 3, padding='same', name='conv{}-{}'.format(block, i), use_bias=False)(block_input)
            else:
                conv2d = Conv2D(filters, 3, padding='same', name='conv{}-{}'.format(block, i), use_bias=False)(conv2d)
            
            conv2d = BatchNormalization(name='batchnorm{}-{}'.format(block, i))(conv2d)
            conv2d = Activation('relu', name='relu{}-{}'.format(block, i))(conv2d)
            
        return conv2d
        
    def build(self, use_cpu=False, print_summary=False):
        vgg16 = VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
        
        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'
        print('='*30+'Loading DeconvNet'+'='*30)
        with tf.device(device):
            inputs = Input(shape=(224, 224, 3),batch_size=self.batch_size)
            print('='*40+'inputs'+'='*40+'\n',inputs)
            conv_block_1 = self.buildConv2DBlock(inputs, 64, 1, 2)
            pool1, pool1_argmax = Lambda(self.max_pool_with_argmax, name='pool1')(conv_block_1) 
            print('='*40+'pool1'+'='*40+'\n',pool1)
            conv_block_2 = self.buildConv2DBlock(pool1, 128, 2, 2)
            pool2, pool2_argmax = Lambda(self.max_pool_with_argmax, name='pool2')(conv_block_2) 
            print('='*40+'pool2'+'='*40+'\n',pool2)
            conv_block_3 = self.buildConv2DBlock(pool2, 256, 3, 3)
            pool3, pool3_argmax = Lambda(self.max_pool_with_argmax, name='pool3')(conv_block_3) 
            print('='*40+'pool3'+'='*40+'\n',pool3)
            conv_block_4 = self.buildConv2DBlock(pool3, 512, 4, 3)
            pool4, pool4_argmax = Lambda(self.max_pool_with_argmax, name='pool4')(conv_block_4) 
            print('='*40+'pool4'+'='*40+'\n',pool4)
            conv_block_5 = self.buildConv2DBlock(pool4, 512, 5, 3)
            pool5, pool5_argmax = Lambda(self.max_pool_with_argmax, name='pool5')(conv_block_5)
            print('='*40+'pool5'+'='*40+'\n',pool5)
            fc6 = Conv2D(512, 7, use_bias=False, padding='valid', name='fc6')(pool5) #4096
            fc6 = BatchNormalization(name='batchnorm_fc6')(fc6)
            fc6 = Activation('relu', name='relu_fc6')(fc6)
            print('='*40+'fc6'+'='*40+'\n',fc6)            
            fc7 = Conv2D(512, 1, use_bias=False, padding='valid', name='fc7')(fc6)   #4096
            fc7 = BatchNormalization(name='batchnorm_fc7')(fc7)
            fc7 = Activation('relu', name='relu_fc7')(fc7)
            print('='*40+'fc7'+'='*40+'\n',fc7)            
            x = Conv2DTranspose(512, 7, use_bias=False, padding='valid', name='deconv-fc6')(fc7)
            x = BatchNormalization(name='batchnorm_deconv-fc6')(x)
            x = Activation('relu', name='relu_deconv-fc6')(x)
            print('='*40+'relu_deconv-fc6'+'='*40+'\n',x)            
            x = MaxUnpoolWithArgmax(pool5_argmax, name='unpool5')(x)
            x.set_shape(conv_block_5.get_shape())
            print('='*40+'unpool5'+'='*40+'\n',x)
            x = Conv2DTranspose(512, 3, use_bias=False, padding='same', name='deconv5-1')(x)
            x = BatchNormalization(name='batchnorm_deconv5-1')(x)
            x = Activation('relu', name='relu_deconv5-1')(x)  
            print('='*40+'relu_deconv5-1'+'='*40+'\n',x)
            x = Conv2DTranspose(512, 3, use_bias=False, padding='same', name='deconv5-2')(x)
            x = BatchNormalization(name='batchnorm_deconv5-2')(x)
            x = Activation('relu', name='relu_deconv5-2')(x)  
            print('='*40+'relu_deconv5-2'+'='*40+'\n',x)
            x = Conv2DTranspose(512, 3, use_bias=False, padding='same', name='deconv5-3')(x)
            x = BatchNormalization(name='batchnorm_deconv5-3')(x)
            x = Activation('relu', name='relu_deconv5-3')(x)  
            print('='*40+'relu_deconv5-3'+'='*40+'\n',x)
            x = MaxUnpoolWithArgmax(pool4_argmax, name='unpool4')(x)
            x.set_shape(conv_block_4.get_shape())
            print('='*40+'unpool4'+'='*40+'\n',x)

            x = Conv2DTranspose(512, 3, use_bias=False, padding='same', name='deconv4-1')(x)
            x = BatchNormalization(name='batchnorm_deconv4-1')(x)
            x = Activation('relu', name='relu_deconv4-1')(x)  
            print('='*40+'relu_deconv4-1'+'='*40+'\n',x)
            x = Conv2DTranspose(512, 3, use_bias=False, padding='same', name='deconv4-2')(x)
            x = BatchNormalization(name='batchnorm_deconv4-2')(x)
            x = Activation('relu', name='relu_deconv4-2')(x)  
            print('='*40+'relu_deconv4-2'+'='*40+'\n',x)
            x = Conv2DTranspose(256, 3, use_bias=False, padding='same', name='deconv4-3')(x)
            x = BatchNormalization(name='batchnorm_deconv4-3')(x)
            x = Activation('relu', name='c3')(x)  
            print('='*40+'relu_deconv4-3'+'='*40+'\n',x)
            x = MaxUnpoolWithArgmax(pool3_argmax, name='unpool3')(x)
            x.set_shape(conv_block_3.get_shape())
            print('='*40+'unpool3'+'='*40+'\n',x)
            x = Conv2DTranspose(256, 3, use_bias=False, padding='same', name='deconv3-1')(x)
            x = BatchNormalization(name='batchnorm_deconv3-1')(x)
            x = Activation('relu', name='relu_deconv3-1')(x)
            print('='*40+'relu_deconv3-1'+'='*40+'\n',x)  
            x = Conv2DTranspose(256, 3, use_bias=False, padding='same', name='deconv3-2')(x)
            x = BatchNormalization(name='batchnorm_deconv3-2')(x)
            x = Activation('relu', name='relu_deconv3-2')(x)  
            print('='*40+'relu_deconv3-2'+'='*40+'\n',x)
            x = Conv2DTranspose(128, 3, use_bias=False, padding='same', name='deconv3-3')(x)
            x = BatchNormalization(name='batchnorm_deconv3-3')(x)
            x = Activation('relu', name='relu_deconv3-3')(x)  
            print('='*40+'relu_deconv3-3'+'='*40+'\n',x)
            x = MaxUnpoolWithArgmax(pool2_argmax, name='unpool2')(x)
            x.set_shape(conv_block_2.get_shape())
            print('='*40+'unpool2'+'='*40+'\n',x)
            x = Conv2DTranspose(128, 3, use_bias=False, padding='same', name='deconv2-1')(x)
            x = BatchNormalization(name='batchnorm_deconv2-1')(x)
            x = Activation('relu', name='relu_deconv2-1')(x)  
            print('='*40+'relu_deconv2-1'+'='*40+'\n',x)
            x = Conv2DTranspose(64, 3, use_bias=False, padding='same', name='deconv2-2')(x)
            x = BatchNormalization(name='batchnorm_deconv2-2')(x)
            x = Activation('relu', name='relu_deconv2-2')(x)  
            print('='*40+'relu_deconv2-2'+'='*40+'\n',x)
            x = MaxUnpoolWithArgmax(pool1_argmax, name='unpool1')(x)
            x.set_shape(conv_block_1.get_shape())
            print('='*40+'unpool1'+'='*40+'\n',x)
            x = Conv2DTranspose(64, 3, use_bias=False, padding='same', name='deconv1-1')(x)
            x = BatchNormalization(name='batchnorm_deconv1-1')(x)
            x = Activation('relu', name='relu_deconv1-1')(x)  
            print('='*40+'relu_deconv1-1'+'='*40+'\n',x)
            x = Conv2DTranspose(64, 3, use_bias=False, padding='same', name='deconv1-2')(x)
            x = BatchNormalization(name='batchnorm_deconv1-2')(x)
            x = Activation('relu', name='relu_deconv1-2')(x)              
            print('='*40+'relu_deconv1-2'+'='*40+'\n',x)
            output = Conv2DTranspose(self.labels, 1, activation='softmax', padding='same', name='output')(x)
            print('='*40+'output'+'='*40+'\n',output)
            self.model = Model(inputs=inputs, outputs=output)
            vgg16 = VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
            
            if print_summary:
                print(self.model.summary())
            
            for layer in self.model.layers:
                if layer.name.startswith('conv'):
                    block = layer.name[4:].split('-')[0]
                    depth = layer.name[4:].split('-')[1]
                    # apply vgg16 weights without bias
                    layer.set_weights([vgg16.get_layer('block{}_conv{}'.format(block, depth)).get_weights()[0]])
            
            #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            #rmsprop=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
            #adagrad=Adagrad(lr=0.01, epsilon=1e-06)
            adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
            self.model.compile(optimizer=adadelta,
                          loss='categorical_crossentropy',
                          metrics=['accuracy', 'mse'])
            #plot_model(self.model,to_file="model.png",show_shapes=True)
            
