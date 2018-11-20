from DeconvNet import DeconvNet
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import numpy as np
import cv2
import sys
import random
labels=4
batch_size=4
sys.setrecursionlimit(10000)
input_size=(224,224,3)

deconvNet = DeconvNet(batch_size=batch_size,labels=labels)
deconvNet.train(epochs=100, steps_per_epoch=1600)
deconvNet.load()

def random_crop_or_pad(image, truth, size=(input_size[0], input_size[1])):
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


images = np.zeros((batch_size, input_size[0], input_size[1], input_size[2]))
truths = np.zeros((batch_size, input_size[0], input_size[1], labels))
test_txt = open('data/train.txt').readlines()
for i in range(4):
    random_line = random.choice(test_txt)
    image_file = random_line.split(' ')[0]
    truth_file = random_line.split(' ')[1]
    image = np.float32(cv2.imread(image_file)/255.0)
    truth_mask = cv2.imread(truth_file[:-1], cv2.IMREAD_GRAYSCALE)
    truth_mask[truth_mask == 255] = 0 # replace no_label with background  
    images[i], truth = random_crop_or_pad(image, truth_mask, input_size)
    truths[i] = (np.arange(labels) == truth[...,None]).astype(int) # encode to one-hot-vector

prediction = deconvNet.predict((images))
prediction = np.argmax(prediction,3)
prediction=prediction*45
images=images*255
org=Image.fromarray(images[1].astype('uint8'))
im = Image.fromarray(prediction[1].astype('uint8'))
im.save("result.jpeg")
org.save("org.jpeg")
