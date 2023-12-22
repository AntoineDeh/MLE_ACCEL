import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.regularizers import l2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_POINTS = 100


def scale_sequences(sequences):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(sequences)
    return sequences


seed = int(time.time())
np.random.seed(seed)
tf.random.set_seed(seed)

dataset_to_load = input("Specify the name of the dataset to load: ")
raw_dataset = pd.read_csv('datasets/' + dataset_to_load + '.csv').values

E_raw_dataset = raw_dataset[:, :-1]
Y_raw_dataset = raw_dataset[:, -1]

# Vérification si la longueur de raw_dataset est un multiple de 200
if len(raw_dataset) % DATA_POINTS != 0:
    raise ValueError(f"Le nombre de lignes dans le fichier CSV n'est pas un multiple de {DATA_POINTS}.")

scaler = MinMaxScaler(feature_range=(0, 1))
E_raw_dataset = scaler.fit_transform(E_raw_dataset)

# Division du dataset en sous-tableaux
E_dataset = np.array_split(E_raw_dataset, len(raw_dataset) // DATA_POINTS)
Y_dataset = np.array_split(Y_raw_dataset, len(raw_dataset) // DATA_POINTS)

# Vérifier que toutes les étiquettes sont identiques dans chaque lot
for lot in Y_dataset:
    if np.any(lot != lot[0]):
        raise ValueError("Toutes les étiquettes ne sont pas identiques dans un lot de 100.")

# Réduire chaque lot à une seule étiquette
Y_dataset = [lot[0] for lot in Y_dataset]

# Conversion en numpy arrays
E_dataset = np.asarray(E_dataset)
Y_dataset = np.asarray(Y_dataset)

E_train_dataset, E_test_dataset, Y_train_dataset, Y_test_dataset = train_test_split(E_dataset, Y_dataset, test_size=0.2,
                                                                                    random_state=seed)
# non pertinent pour le moment
l2_regularizer = l2(0.01)
# l2_regularizer = None

# L'ajout de EarlyStopping et ReduceLROnPlateau peut être intéressant
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=40,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=20,
    verbose=1,
    mode='min',
    min_lr=0.001
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(DATA_POINTS, activation='relu', input_shape=(E_train_dataset.shape[1], E_train_dataset.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=l2_regularizer),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=l2_regularizer),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.2, weight_decay=0.0)
rms_optimizer = tf.keras.optimizers.RMSprop(momentum=0.1)

metrics_accuracy = ['accuracy']
# non pertinent pour le moment
metrics_binary_accuracy = [tf.keras.metrics.BinaryAccuracy()]
metrics_auc = [tf.keras.metrics.AUC()]
# non pertinent pour le moment
metrics_binary_accuracy_auc = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
# Des pics étaient observés avec metrics_auc et dataset_v1, donc nous avons essayer precision et recall mais résultats
# plus mauvais. Nous n'avons pas encore essayé de modifié les paramètres de ces fonction
metric_precision_recall = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

model.compile(optimizer=rms_optimizer,
              loss='binary_crossentropy',
              metrics=metrics_auc)

# model.summary()

history = model.fit(E_train_dataset,
                    Y_train_dataset,
                    shuffle=False,
                    epochs=200,
                    batch_size=20,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])

# evaluate the model
scores = model.evaluate(E_test_dataset, Y_test_dataset)
print("\nEvaluation sur le test data %s: %.2f - %s: %.2f%% " % (
    model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

# plot figure
plt.figure()
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('Model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('Model precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
# Plot training & validation loss values

# save the model
model_name = input("Do you want to save the model? (File name/n) ")
if model_name != "n" and model_name != "":
    os.makedirs('models/' + model_name, exist_ok=True)
    model.save('models/' + model_name + '/' + model_name + '.h5')

    model_structure = model.to_json()
    with open('models/' + model_name + '/' + model_name + '.json', "w") as json_file:
        json_file.write(model_structure)

    plt.figure()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model auc')
    plt.ylabel('Auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('models/' + model_name + '/' + model_name + '_auc_plot.png')

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('models/' + model_name + '/' + model_name + '_loss_plot.png')

    print("Model saved.")
