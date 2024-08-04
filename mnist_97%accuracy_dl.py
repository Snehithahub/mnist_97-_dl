import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np


print(tf.__version__)
m=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=m.load_data()
x_train=x_train/255.0
x_test=x_test/255.0


model=tf.keras.models.Sequential(
    [tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)])


p=model(x_train[:1]).numpy()
print(p)
pro=tf.nn.softmax(p).numpy()
predicted_classes = np.argmax(pro, axis=1)
accuracy = np.mean(predicted_classes == y_train)
print("Pre-training accuracy:", accuracy)


l=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("loss",l(y_train[2:3], p).numpy())


model.compile(optimizer='adam',
              loss=l,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6,verbose=1)


test_loss,test_Accuracy=model.evaluate(x_test,y_test)
print(test_loss,test_Accuracy)
