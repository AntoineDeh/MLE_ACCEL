import os
import tkinter as tk
from tkinter import filedialog, messagebox

def build_model():
    model = Sequential()
    model.add(Dense(64,activation='relu',input_shape=(Xtrain.shape[1],)))
    model.add(Dense(128,activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    save_model(model)

def save_model(model):
    file_path = filedialog.asksaveasfilename(title="Enregistrer le modèle", defaultextension=".h5",
                                             filetypes=[("H5 Files", "*.h5")])
    if file_path:
        try:
            model.save(file_path)
            print("Succès", "Modèle enregistré avec succès.")
        except Exception as e:
            print("Erreur", f"Erreur lors de l'enregistrement du modèle: {e}")

     #enregistre dans le dossier model