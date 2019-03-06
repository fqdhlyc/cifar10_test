import datetime
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Lambda,Dropout,Conv2D,MaxPooling2D
import keras
import numpy as np
import random
import sys, os
#import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def carlini_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter):
    """
    :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
    :params input_range_type: the expected input range, {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
    """

    model = Sequential()

    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Convert to [-0.5,0.5] by x-0.5.
        scaler = lambda x: x-0.5
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5].
        # Don't need to do scaling for carlini models, as it is the input range by default.
        scaler = lambda x: x
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [-0.5,0.5] by x/2.
        scaler = lambda x: x/2

    model.add(Lambda(scaler, input_shape=input_shape))
    model.add(Lambda(pre_filter, output_shape=input_shape))

    model.add(Conv2D(nb_filters, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(nb_filters*2, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters*2, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(nb_denses[0]))
    model.add(Activation('relu'))
    model.add(Dense(nb_denses[1]))
    model.add(Activation('relu'))
    model.add(Dense(nb_denses[2]))

    model.add(Activation('softmax'))

    return model

def carlini_cifar10_model(logits=True, input_range_type=1, pre_filter=lambda x:x):
    
    input_shape=(32, 32, 3)
    nb_filters = 64
    nb_denses = [256,256,10]
    return carlini_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter)

def get_test_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)
    del X_train, y_train
    return X_test, Y_test

def getacc(a,b):
    a = np.argmax(a,axis=1)
    b = np.argmax(b,axis=1)
    vec = (a==b)
    acc = np.sum(vec)/float(len(b))
    return acc

def getmatch(pred,label):
    pred = pred[0]
    Y_pred_class = np.argmax(pred, axis = 0)
    Y_label_class = np.argmax(label, axis = 0)
    return Y_pred_class == Y_label_class

def select_seeds_by_class(x,y,model):
    x_seeds = []
    y_seeds = []
    count = [0]*10
    counter = []
    for i in range(10000):
        temp = int(np.argmax(y[i], axis = 0))
        if (count[temp]==10):
            continue
        elif(sum(count)==100):
            break
        else:
            test = x[i]
            test = test.reshape((1,32,32,3))
            result = model.predict(test)
            if (getmatch((result),y[i])):
                #print(temp)
                counter.append(i)
                count[temp]+=1
                x_seeds.append(x[i])
                y_seeds.append(y[i])
    x = np.array(x_seeds)
    y = np.array(y_seeds)
    return x,y,counter

if __name__ == '__main__':
    cifar = carlini_cifar10_model()
    X_test,Y_test = get_test_dataset()
    cifar.load_weights('cifar10.h5')
    cifar.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    
    x_seeds,y_seeds,counter = select_seeds_by_class(X_test,Y_test,cifar)

    Y_pred = cifar.predict(x_seeds)

    acc_all = getacc(Y_pred,y_seeds)
    print('Test accuracy on legitimate examples %.4f' % (acc_all))

    ad_x_index = []
    ad_ep = []
    
    ad_vector=[]

    ad_vector_i = []
    
    test_vec= np.zeros((10000,32,32,3))
    val_vec = np.zeros((10000))
    si = True

    for i in range(len(x_seeds)):
        print("seed: %d"%i)
        print(datetime.datetime.now())
        si = True
        for z in range(26):
            ep = 0.05+(z*0.01)
            ep_attack = ep*(np.ones((32,32,3)))
            #x_upper = x_nat[i]+ep_attack
            #x_lower = x_nat[i]-ep_attack
            print("ep: %f"%ep)
            #test_vec = np.zeros((1000,28,28,1))
            #val_vec = np.zeros((1000))
            for j in range(10000):

                attack = np.random.uniform(-ep,ep,size=(32,32,3))
                result1 = x_seeds[i]+attack

                result1 = np.clip(result1,0,1)
                test_vec[j] = result1
                val_vec[j] = np.argmax(y_seeds[i],axis=0)

            y_predicted = np.argmax(cifar.predict(test_vec), axis=1)
            #val = np.argmax(val_vec,axis=1)
            difference = np.where(y_predicted!=val_vec)
            difference = difference[0]
            if(len(difference)>0 and si):
                ad_vector.append(test_vec[difference[0]])
                ad_vector_i.append(ep)
                si = False
            for k in difference:
                ad_x_index.append(i)
                ad_ep.append(ep)


            
    np.save('x_seeds',x_seeds)
    np.save('y_seeds',y_seeds)
    np.save('ad_x_index',np.array(ad_x_index))
    np.save('ad_ep',np.array(ad_ep))
    np.save('ad_vector',np.array(ad_vector))
    np.save('ad_vector_ep',np.array(ad_vector_i))
