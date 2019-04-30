# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras import backend as K
from keras import initializers
from keras.datasets import mnist
import collections

# fix random seed for reproducibility
seed = 10
np.random.seed(seed)

# Import data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Do a custom train/test split
x1 = np.concatenate((X_train, X_test))
y1 = np.concatenate((y_train, y_test))

train_size = 0.3
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, train_size=train_size)

# Subset data to only include seven and one
train_filter = np.where((y_train1 == 7) | (y_train1 == 1))
test_filter = np.where((y_test1 == 7) | (y_test1 == 1))

X_train1, y_train1 = X_train1[train_filter], y_train1[train_filter]
X_test1, y_test1 = X_test1[test_filter], y_test1[test_filter]

# Recode 7 as zero for sigmoid function in model
y_train1[y_train1 == 7] = 0
y_test1[y_test1 == 7] = 0

# Count recoded sevens and ones to check balance of the target variable
print("Y_train class balance: ", collections.Counter(y_train1))
print("Y_test class balance: ", collections.Counter(y_test1))

# Reshape data to be [samples][pixels][width][height]
X_train1 = X_train1.reshape(X_train1.shape[0], 28, 28, 1).astype('float32')
X_test1 = X_test1.reshape(X_test1.shape[0], 28, 28, 1).astype('float32')

# Rescale inputs from 0-255 to 0-1
X_train1 = X_train1 / 255
X_test1 = X_test1 / 255

# Dimensions of images
img_width = 28
img_height = 28

# Initialize structure parameters
num_train = X_train1.shape[0]
num_test = X_test1.shape[0]

epochs = 10
batch_size = 100

# Define Input shape
input_shape = (img_width, img_height, 1)

# Brier Score function for Keras


def brier_score(y_true, y_pred):
    """
    Calculates the Brier Score using Keras backend.
    Arguments: y_true (the true target values), y_pred (the predicted target values)
    Output: Brier Score (float)
    """
    n = K.sum(y_pred) / K.mean(y_pred)
    return K.sum(K.pow((y_pred-y_true), 2)) / n

# Model Definition


model = Sequential()
model.add(Conv2D(4, (5, 5), input_shape=input_shape, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(3, (5, 5), activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4, kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', brier_score])

# Fit the model
H = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=epochs, batch_size=batch_size)

# Predict from the model
y_pred1 = model.predict(X_test1)
print("Brier Score: ", brier_score_loss(y_test1, y_pred1))
# Final evaluation of the model
scores = model.evaluate(X_test1, y_test1, verbose=0)

# Show summary of network parameters
print("Model Summary: ", model.summary())


model.save_weights('mod1_weights.h5')

# Calculate new target variable for Model 3
new_target = np.zeros(y_test1.shape[0])
for i in range(len(y_test1)):
    new_target[i] = np.abs(y_pred1[i]-y_test1[i])

# Export data to be used in Model 3
np.save('new_y.npy', new_target)
np.save('new_x.npy', X_test1)
np.save('mod1_y_preds', y_pred1)
np.save('mod1_y_test', y_test1)
np.save('mod1_y_train', y_train1)
np.save('mod1_x_train', X_train1)

# Plot 1: Train/Test Loss, Accuracy, and Brier Score
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="Train Accuracy")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="Validation Accuracy")
plt.plot(np.arange(0, epochs), H.history['brier_score'], label='Brier Score')
plt.plot(np.arange(0, epochs), H.history['val_brier_score'], label='Validation Brier Score')
plt.title("Model 1: Loss, Accuracy, and Brier Score on MNIST Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy/Brier Score")
plt.legend(loc="center right")
plt.savefig("lossAccBrierPlot1.png")

# Plot 2: Calibration Plot
nn_y1, nn_x1 = calibration_curve(y_test1, y_pred1, n_bins=11)
fig, ax = plt.subplots()
plt.plot(nn_x1, nn_y1, marker='o', linewidth=1, label='Model 1 NN')
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('MNIST Data Calibration with Model 1')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('True Probability per Bin')
plt.legend(loc='upper left')
plt.savefig("calibrationPlot1.png")

# Plot 3: 2-class Density Plot
plt.figure()
for i in [0, 1]:
    subset = [y_test1 == i]
    sns.distplot(y_pred1[subset], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label=i)
plt.legend(prop={'size': 16}, title='Class')
plt.title('Model 1: Density Plot of Classes')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.savefig("densityPlot1.png")

# Clear backend weights and initialization
K.clear_session()

# Brier Score:  0.006943680989451738
# loss: 0.0291 - acc: 0.9913 - brier_score: 0.0070 - val_loss: 0.0310 - val_acc: 0.9910 - val_brier_score: 0.0069