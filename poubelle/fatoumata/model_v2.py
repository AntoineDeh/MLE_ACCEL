import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Importation des modules spécifiques de TensorFlow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

# Initialisation de la graine aléatoire pour la reproductibilité
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# Chargement et affichage des données
df = pd.read_csv('SensorTile_Log_N000.csv')
print(df)
print(df.isna().sum())
df.plot()

# Préparation des étiquettes
df['labels'] = 'non balencement'
df.loc[16:2962, 'labels'] = 'balencement'  # Intervalles de balancement
df.loc[list(range(0, 15)) + list(range(2963, 2974)), 'labels'] = 'non balencement'
print(df)

# Sélection des caractéristiques et des étiquettes
x = df.iloc[:, 1:3]  # Sélection des colonnes AccX et AccY
y = df['labels'].values.reshape(-1, 1)  # Conversion et remodelage des étiquettes

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

# Encodage des étiquettes
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Préparation des données pour l'entraînement
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Construction du modèle
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])
model.summary()

# Compilation du modèle
opt = SGD(learning_rate=0.001, momentum=0.2)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement avec validation et arrêt anticipé
callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train_encoded, validation_split=0.15, epochs=500, batch_size=30, callbacks=[callback], use_multiprocessing=True)

# Visualisation de la perte et de la précision
plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
plt.title('Loss Evolution')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
plt.title('Accuracy Evolution')
plt.legend()
plt.show()

# Évaluation du modèle
scores = model.evaluate(X_test, y_test_encoded)
print("\nÉvaluation sur le test data %s: %.2f - %s: %.2f%% " % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

# Analyse temporelle des signaux
plt.figure(figsize=(12, 18))
for i, axis in enumerate(['AccX [mg]', 'AccY [mg]', 'AccZ [mg]'], start=1):
    plt.subplot(3, 1, i)
    plt.plot(df['T [ms]'], df[axis], label=axis)
    plt.title(f'Signal {axis} dans le domaine temporel')
    plt.xlabel('Temps (ms)')
    plt.legend()
plt.tight_layout()
plt.show()
