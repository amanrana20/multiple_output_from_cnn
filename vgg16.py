from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import theano


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
conv_1 = Convolution2D(64, 3, 3, activation='relu')
model.add(conv_1)
model.add(ZeroPadding2D((1,1)))
conv_2 = Convolution2D(64, 3, 3, activation='relu')
model.add(conv_2)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
conv_3 = Convolution2D(128, 3, 3, activation='relu')
model.add(conv_3)
model.add(ZeroPadding2D((1,1)))
conv_4 = Convolution2D(128, 3, 3, activation='relu')
model.add(conv_4)
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
conv_5 = Convolution2D(256, 3, 3, activation='relu')
model.add(conv_5)
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))


model.load_weights('vgg16_weights.h5')
#    model.summary()

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    for i in range(6):
        model.pop()

    for i in range(-30, 0):
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = model.predict(im)
        IMAGE = None
        images = out[0, :, :, :]
        print images.shape
        for j in range(10):
            image = images[j, :, :]
            if IMAGE is None:
                IMAGE = image
            else:
                IMAGE = np.concatenate((IMAGE, image), axis=1)

#        cv2.imshow('image', IMAGE)
        name = '10_images_in_each_layer/vgg16_level{}.jpg'.format(-i)
#        print name
        cv2.imwrite(name, IMAGE)
#        cv2.waitKey(500)
#        print IMAGE.shape
#        print

        model.pop()
