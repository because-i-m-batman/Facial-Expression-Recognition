from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization
from tensorflow.keras import backend as K 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


input_shape = (48,48,3)

#model Architecture

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Conv2D(256, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(7))
model.add(Activation('softmax'))

opt = Adam(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

#Model Inputs

train_datagen = ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization = True,
        rotation_range = 10,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range = [0.2,1.0],
        zoom_range = 0.2,
        validation_split = 0.05)

test_datagen = ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization = True)  

#spliting the data into validation and training data using ImageDatagenerator
train_generator = train_datagen.flow_from_directory('Your Training Data Path',target_size = (48,48),batch_size = 32,shuffle = True,seed = 42,subset = 'training')
valid_generator = train_datagen.flow_from_directory('Your Training Data Path',target_size = (48,48),batch_size = 32,shuffle = True,seed = 42,subset = 'validation')

#Checkpoints for model using minimum validation loss

#Note:Give path for keras model format
checkpoint = ModelCheckpoint('Model Path where you want to store the best model', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
print(model.summary())

#Model Compiling
print(train_generator.samples)
print(valid_generator.samples)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples//32,
        epochs=300,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples//32,
        callbacks = callbacks_list)

#plot the curves for accuracy and loss 
import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
