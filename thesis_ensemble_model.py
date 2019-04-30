# Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.calibration import calibration_curve
import seaborn as sns
from sklearn.metrics import brier_score_loss
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras import initializers
from keras import backend as K
import collections

# Import data from Model 1
X_train2 = np.load('mod1_x_train.npy')
X_test2 = np.load('new_x.npy')
y_train2 = np.load('mod1_y_train.npy')
y_test2 = np.load('mod1_y_test.npy')

# Show the shape of each numpy array
print("Shape of X_train: ", X_train2.shape[0])
print("Shape of Y_train: ", y_train2.shape[0])
print("Shape of X_test: ", X_test2.shape[0])
print("Shape of Y_test: ", y_test2.shape[0])

# Count balance between recoded sevens and ones of target variable
print("Y_train class balance: ", collections.Counter(y_train2))
print("Y_test class balance: ", collections.Counter(y_test2))

# Reshape data to be [samples][pixels][width][height]
X_train2 = X_train2.reshape(X_train2.shape[0], 28, 28, 1).astype('float32')
X_test2 = X_test2.reshape(X_test2.shape[0], 28, 28, 1).astype('float32')

# Dimensions of images
img_width = 28
img_height = 28

# Initialize structure parameters
num_train = X_train2.shape[0]
num_test = X_test2.shape[0]

epochs = 5
batch_size = 100
num_ensemble = 5

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


def modelRun():
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
    model.fit(X_train2, y_train2, validation_data=(X_test2, y_test2), epochs=epochs, batch_size=batch_size, verbose=0)

    # Predict from the model
    y_pred2 = model.predict(X_test2)
    # Final evaluation of the model
    score = model.evaluate(X_test2, y_test2, verbose=0)
    print("Evaluation Scores: ", score)

    # Show summary of network parameters
    print("Model Summary: ", model.summary())

    model.save_weights('mod2_weights.h5')

    # Return the evaluation scores and the output predictions
    return score, y_pred2

# Run ensemble of models and collect scores and predictions


all_scores = {'Loss': list(), 'Accuracy': list(), 'Brier Score': list()}
all_y_pred2 = list()
for i in range(num_ensemble):
    print("Training Network ", i)
    temp_score, temp_y_pred = modelRun()
    if i == 0:
        for j in temp_y_pred:
            all_y_pred2.append(list(j))
    else:
        for j in range(len(temp_y_pred)):
            all_y_pred2[j].append(temp_y_pred[j][0])
    for index, key in enumerate(all_scores):
        all_scores[key].append(temp_score[index])

    # Clear backend weights and initialization
    K.clear_session()

print("Scores from all networks: ", all_scores)

# Calculate mean of all scores
mean_scores = {'Loss': 0, 'Accuracy': 0, 'Brier Score': 0}
for i in all_scores:
    mean_scores[i] = np.mean(all_scores[i])

print("Mean scores from all networks (averaged over the # of networks): ", mean_scores)

# Calculate mean of all predictions
mean_y_pred2 = np.zeros(len(all_y_pred2))
for i in range(len(all_y_pred2)):
    mean_y_pred2[i] = np.mean(all_y_pred2[i])

# Brier Score for all the ensembles
print("Brier score of averaged output probabilities: ", brier_score_loss(y_test2, mean_y_pred2))

# Plot 1: Calibration Plot
plt.style.use("ggplot")
nn_y2, nn_x2 = calibration_curve(y_test2, mean_y_pred2, n_bins=11)
fig, ax = plt.subplots()
plt.plot(nn_x2, nn_y2, marker='o', linewidth=1, label='Model 2 NN')
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('MNIST Data Calibration with Model 2')
ax.set_xlabel('Predicted Probability')
ax.set_xlim((0, 1))
ax.set_ylabel('True Probability per Bin')
plt.legend(loc='upper left')
plt.savefig("calibrationPlot2.png")

# Plot 2: 2-class Density Plot
fig, ax = plt.subplots()
for i in [0, 1]:
    subset = [y_test2 == i]
    sns.distplot(mean_y_pred2[subset], hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label=i)
plt.legend(prop={'size': 16}, title='Class')
fig.suptitle('Model 2: Density Plot of Classes')
ax.set_xlabel('Probability')
ax.set_xlim((-.1, 1.1))
ax.set_ylabel('Density')
plt.savefig("densityPlot2.png")

'''
Scores from all networks:  {'Loss': [0.6926070482792754, 0.05515977530615833, 0.06627609057692498, 0.06213231555710479, 0.6926063058213434], 'Accuracy': [0.5164464436406528, 0.9845375316277762, 0.981819885671446, 0.9831318526848468, 0.5164464436406528], 'Brier Score': [0.24972999824566913, 0.0131159654725679, 0.01577364374200887, 0.014603354033927905, 0.2497295860709765]}
Mean scores from all networks (averaged over the # of networks):  {'Loss': 0.3137563071081614, 'Accuracy': 0.796476431453075, 'Brier Score': 0.10859050951303006}
Brier score of averaged output probabilities:  0.05365545483414875
'''