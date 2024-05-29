import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import pandas as pd
import matplotlib.pyplot as plt

path_train ='train'
path_test = 'test1'

if os.path.exists(path_train):
    filenames = os.listdir(path_train)
else:
    print(f"Directory does not exist: {path_train}")
categories = []

for f in filenames:
    category = f.split('.')[0]
    if category == 'dog':
        categories.append(1)
        class_num = 1
    else:
        categories.append(0)
        class_num = 0

df=pd.DataFrame({
    'filename':filenames,
    'category':categories
})

img_width, img_height = 128, 128
batch_size = 15
print(df.head())

## generate image

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
df["category"] = df["category"].replace({0:'cat',1:'dog'})
train_df,validate_df = train_test_split(df,test_size=0.20,
  random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=15

print(total_train, total_validate)

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_generator = train_datagen.flow_from_dataframe(
    train_df,
    path_train,
    x_col ='filename', y_col ='category',
    target_size=(img_width, img_height),
    class_mode='binary',
    batch_size=batch_size  # Set this to 'categorical' if you have multiple classes
)


validate_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


validate_generator = train_datagen.flow_from_dataframe(
    validate_df,
    path_train,
    x_col='filename',y_col='category',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Set this to 'categorical' if you have multiple classes
)

path_test = "test1"
if os.path.exists(path_test):
    filenames = os.listdir(path_test)
else:
    print(f"Directory does not exist: {path_test}")
test_df = pd.DataFrame({
    'filename': filenames
})

nb_samples = test_df.shape[0]
test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                 path_test,
                                                 x_col='filename',
                                                 target_size=(img_width,img_height),
                                                 class_mode=None,
                                                 batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization
import random

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_width,img_height,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
callbacks = [earlystop,learning_rate_reduction]


history = model.fit_generator(
    train_generator,
    epochs=3,
    validation_data=validate_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save("model1_catsVSdogs_10epoch.h5")
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)
predicted_classes = pd.DataFrame(predicted_classes)
predicted_classes.replace({0:"cat", 1: "dog"})
print(predicted_classes[:5])