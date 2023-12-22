import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Paramètres de base
DATA_POINTS = 100
seed = np.random.randint(10000)

# Définition des fonctions pour l'interface graphique et l'entraînement
def load_dataset():
    file_path = filedialog.askopenfilename(title="Choisir le dataset", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            global raw_dataset, E_raw_dataset, Y_raw_dataset
            raw_dataset = pd.read_csv(file_path).values
            E_raw_dataset = raw_dataset[:, :-1]
            Y_raw_dataset = raw_dataset[:, -1]
            messagebox.showinfo("Succès", "Dataset chargé avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement du fichier: {e}")

def build_model(input_shape1, input_shape2):
    l2_regularizer = l2(0.01)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(DATA_POINTS, activation='relu', input_shape=(input_shape1, input_shape2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=l2_regularizer),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=l2_regularizer),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    save_model(model)

def save_model(model):
    file_path = filedialog.asksaveasfilename(title="Enregistrer le modèle", defaultextension=".h5",
                                             filetypes=[("H5 Files", "*.h5")])
    if file_path:
        try:
            model.save(file_path)
            messagebox.showinfo("Succès", "Modèle enregistré avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'enregistrement du modèle: {e}")

def train_model():
    try:
        # Vérification et préparation des données
        if len(raw_dataset) % DATA_POINTS != 0:
            raise ValueError(f"Le nombre de lignes dans le fichier CSV n'est pas un multiple de {DATA_POINTS}.")

        scaler = MinMaxScaler(feature_range=(0, 1))
        E_dataset_scaled = scaler.fit_transform(E_raw_dataset)
        E_dataset = np.array_split(E_dataset_scaled, len(raw_dataset) // DATA_POINTS)
        Y_dataset = np.array_split(Y_raw_dataset, len(raw_dataset) // DATA_POINTS)

        for lot in Y_dataset:
            if np.any(lot != lot[0]):
                raise ValueError("Toutes les étiquettes ne sont pas identiques dans un lot de 100.")

        Y_dataset = [lot[0] for lot in Y_dataset]
        E_dataset = np.asarray(E_dataset)
        Y_dataset = np.asarray(Y_dataset)

        # Division en ensembles d'entraînement et de test
        E_train, E_test, Y_train, Y_test = train_test_split(E_dataset, Y_dataset, test_size=0.2, random_state=seed)

        # Construction et entraînement du modèle
        model = build_model(E_train.shape[1], E_train.shape[2])
        history = model.fit(E_train, Y_train, epochs=200, batch_size=20, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

        # Affichage des résultats
        plot_graphs(history)
        messagebox.showinfo("Succès", "Entraînement terminé avec succès.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors de l'entraînement du modèle: {e}")



def plot_graphs(history):
    fig = Figure(figsize=(10, 8), dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(history.history['accuracy'], label='Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper left')

    ax2.plot(history.history['loss'], label='Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper left')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Configuration de l'interface graphique
root = tk.Tk()
root.title("Gestionnaire de Modèle ML")

load_dataset()

build_model() # save_model() is including

train_model() #plot_graphs() is including

# Configuration des callbacks pour l'entraînement
early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, verbose=1, mode='min', min_lr=0.001)

# Lancement de l'interface graphique
root.mainloop()
