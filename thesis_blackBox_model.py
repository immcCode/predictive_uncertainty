# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.special import logit, expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras import backend as K
from keras import initializers

# Import data
X = np.load('new_x.npy')
y = np.load('new_y.npy')

mod1_y_pred = np.load('mod1_y_preds.npy')
mod1_y_test = np.load('mod1_y_test.npy')

# Use logit function to map y to all real numbers for regression
y = logit(y)

# Reshape data to be [samples][pixels][width][height]
X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')

# Dimensions of images
img_width = 28
img_height = 28

# Initialize structure parameters
num_train = X.shape[0]

epochs = 10
batch_size = 100

# Define Input shape
input_shape = (img_width, img_height, 1)

# Root Mean Squared Error Function


def rmse(y_true, y_pred):
    """
    Calculates the Root Mean Squared Error using Keras backend.
    Arguments: y_true (the true target values), y_pred (the predicted target values)
    Output: Root Mean Squared Error (float)
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Model Definition


model = Sequential()
model.add(Conv2D(16, (5, 5), input_shape=input_shape, activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(800, kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('relu'))
model.add(Dense(1, kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(Activation('linear'))

model.compile(loss=rmse,
              optimizer='adam',
              metrics=[rmse])

# Fit the model
H = model.fit(X, y, epochs=epochs, batch_size=batch_size)
mod3_y_pred = model.predict(X, batch_size=batch_size)

# Use the logistic function to remap predicted y to 0-1 range
mod3_y_pred = expit(mod3_y_pred)

# Combine predicted deviation to base model y_pred
new_y = np.zeros(y.shape[0])
for i in range(len(y)):
    if mod3_y_pred[i] >= .5:
        new_y[i] = .5
    elif mod3_y_pred[i] < .5 and mod1_y_pred[i] < .5:
        new_y[i] = np.mean((mod3_y_pred[i], mod1_y_pred[i]))
    elif mod3_y_pred[i] < .5 and mod1_y_pred[i] >=.5:
        new_y[i] = np.mean((1-mod3_y_pred[i], mod1_y_pred[i]))

# Score new values
print("Brier score of adjusted predictions: ", brier_score_loss(mod1_y_test, new_y))

# Show summary of network parameters
print("Model Summary: ", model.summary())

model.save_weights('mod3_weights.h5')

# Plot 1: Loss Plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="RMSE Loss")
plt.title("Model 3: Root Mean Squared Error Loss on MNIST Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="center right")
plt.savefig("lossPlot3.png")

# Plot 2: Calibration Plot
nn_y, nn_x = calibration_curve(mod1_y_test, new_y, n_bins=11)
fig, ax = plt.subplots()
plt.plot(nn_x, nn_y, marker='o', linewidth=1, label='Model 3 NN')
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('MNIST Data Calibration with Model 3')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('True Probability per Bin')
plt.legend(loc='upper left')
plt.savefig("calibrationPlot3.png")

# Plot 3: 2-class Density Plot
plt.figure()
for i in [0, 1]:
    subset = [mod1_y_test == i]
    sns.distplot(new_y[subset], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label=i)
plt.legend(prop={'size': 16}, title='Class')
plt.title('Model 3: Density Plot of Classes')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.savefig("densityPlot3.png")

# Plot 4: Predicted Probabilities by Class Plot
fig, ax = plt.subplots()
mod3_plot4_colors = ['blue', 'orange']
for i in [0, 1]:
    subset = [mod1_y_test == i]
    plt.scatter(mod1_y_pred[subset], new_y[subset], c=mod3_plot4_colors[i], label=i)
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Model 1 and 3 Predicted Probabilities by Class')
ax.set_xlabel('Model 1 Predicted Probabilities')
ax.set_ylabel('Model 3 Adjusted Probabilities')
plt.legend(title='Class', loc='lower right')
plt.savefig('predProbPlot3.png')

# Create csv of y_test and y_pred values over Models 1 and 3
mod1_y_pred = np.reshape(mod1_y_pred, len(mod1_y_test))
mod3_y_pred = np.reshape(mod3_y_pred, len(mod3_y_pred))
output_csv = np.column_stack((mod1_y_test, mod1_y_pred, new_y, mod3_y_pred))
np.savetxt("y_values.csv", output_csv, delimiter=',', header="model1_y_true,model1_y_pred,model3_adj_y,model3_y_pred")

# Clear backend weights and initialization
K.clear_session()
