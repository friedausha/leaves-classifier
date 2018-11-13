from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import keras.preprocessing.image as image
import numpy as np
import glob
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau,GetBest
import os
from numpy import argmax
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.metrics import accuracy_score


#os.environ['CUDA_VISIBLE_DEVICES']='-1'

sizeImgX=64
sizeImgY=64
numClass=40

def createModel():
    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
    classifier.add(Conv2D(10, (3, 3), input_shape = (sizeImgX, sizeImgY, 3), 
                          activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
#    classifier.add(Conv2D(10, (3, 3), activation = 'relu'))
#    classifier.add(MaxPooling2D(pool_size = (2, 2)))
#    
#    classifier.add(Conv2D(10, (3, 3), activation = 'relu'))
#    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 200, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = numClass, activation = 'softmax'))

    # Compiling the CNN
    classifier.compile(optimizer = optimizers.SGD(lr=1e-4, momentum=0.9) , 
                       loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    classifier.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(sizeImgX,sizeImgY))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]


    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
    
def predict(modelnya,imgFile):
    # Part 3 - Making new predictions
#    test_image = image.load_img(imgFile, target_size=(sizeImgX,sizeImgY))
#    test_image = image.img_to_array(test_image)
#    test_image = np.expand_dims(test_image, axis = 0)
#    result = modelnya.predict_classes(test_image) 
   
    test_image =  load_image(imgFile, True)
    result = modelnya.predict(test_image)
    
    return result    

def loadedModel(filepath):
    model = createModel() 
    model.load_weights(filepath)
    model.compile(optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), 
                  loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print("Created model and loaded weights from file")
    return model


        

image_dir='Daun/train'
generator = image.ImageDataGenerator(validation_split=0.5,rescale=1./255)
training_set = generator.flow_from_directory(image_dir, subset='training',
                                             target_size = (sizeImgX, sizeImgY),                                             
                                             batch_size = 7
                                             ,class_mode='categorical'
                                             ,color_mode="rgb"
                                                
                                             )
test_set = generator.flow_from_directory(image_dir, subset='validation',
                                            target_size = (sizeImgX, sizeImgY),
                                            batch_size =7
                                            ,class_mode='categorical'
                                            ,color_mode="rgb"
                                            )

# checkpoint
#directory = './output/checkpoints/'
#if not os.path.exists(directory):
#    os.makedirs(directory)
#
bestfilepath="weights.best.hdf5"
## Helper: Save the model.
#checkpointer = ModelCheckpoint(
#    filepath = bestfilepath,
##    filepath='./output/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
#    verbose=1,
#    save_best_only=True,mode='max'
#    ,monitor='val_acc'    
#    )

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=15,monitor='val_acc')
# Helper: TensorBoard
#tensorboard = TensorBoard(log_dir='./output/')

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                              patience=15, min_lr=0.001)
model = createModel()
#model.fit_generator(training_set,steps_per_epoch = 8000,
#                    epochs = 25,validation_data = test_set,
#                    validation_steps = 2000,  
#                    callbacks=[checkpointer, early_stopper, tensorboard], 
#                    verbose=1)

#callbacks = [checkpointer, reduce_lr,early_stopper]
callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max'), 
             reduce_lr,early_stopper]
#https://github.com/keras-team/keras/issues/2768-->>GetBest
# callback so that it can store and reset to the best result at the end of training. 
#The weight will only store in the memory no need to write to the disk.
#model.fit_generator(training_set,steps_per_epoch = 300,
#                    epochs = 1000,validation_data = test_set,
#                    validation_steps = 50,  
#                    callbacks=[checkpointer, early_stopper, tensorboard, reduce_lr], 
##                    callbacks=[early_stopper], 
#                    verbose=1)
model.fit_generator(training_set,steps_per_epoch = 300,
                    epochs = 1000,validation_data = test_set,
                    validation_steps = 50,  
                    callbacks=callbacks, 
                    verbose=1)
## save model
model.save(bestfilepath, overwrite=True)  
#print("Daftar Class")                 
dic=training_set.class_indices
inv_map = {v: k for k, v in dic.items()}
#
model=loadedModel(bestfilepath)
scoreSeg = model.evaluate_generator(test_set)
print("Accuracy evaluate_generator = %g"%scoreSeg[1])
pred = model.predict_generator(test_set)
print("Accuracy predict_generator = %g"%accuracy_score(test_set.classes,
                                                       pred.argmax(axis=-1)))

#for imagePath in glob.glob("samplekunci/*.jpg"):
##for imagePath in glob.glob("*.png"):
#    print("Proses : %s"%imagePath)
#    test_image =  load_image(imagePath, True)
##    classes = model.predict_classes(test_image, batch_size=16)
##    inID = classes[0]
##    label = inv_map[inID]
##    print("Image ID: {}, Label: {}".format(inID, label))
##    print classes
#    r = model.predict(test_image, batch_size=16)
#    print r
#    
##    r=predict(model,imagePath)
##    print(r)    
#    y_classes = r.argmax(axis=-1)[0]
##    print(y_classes)
#    label = inv_map[y_classes]
#    print("Name={} Image ID: {}, Label: {} Prosentase={}"
#        .format(imagePath,y_classes, label,r[0][y_classes]))
#

