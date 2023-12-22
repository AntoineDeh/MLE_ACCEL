import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def verify_folder_input(user_input_folder):
    """ Vérifie si le dossier spécifié est valide et retourne le nom du dossier. """
    directory = None
    if user_input_folder in ['input', 'in', 'i']:
        directory = "input"
    elif user_input_folder in ['output', 'out', 'o']:
        directory = "output"
    elif user_input_folder in ['dataset', 'data', 'datasets', 'd']:
        directory = "datasets"
    return directory

def verify_csv_input(user_input_csv, checked_directory):
    """ Vérifie si le fichier CSV spécifié existe dans le dossier donné. """
    test_path = os.path.join(checked_directory, user_input_csv + '.csv')
    return os.path.isfile(test_path), test_path

def add_t_column(csv_df):
    """ Ajoute une colonne de temps incrémentale au DataFrame. """
    csv_df['T [ms]'] = range(0, 20 * len(csv_df), 20)

def plot_data(df, directory):
    """ Effectue des visualisations des données du DataFrame. """
    temps = df['T [ms]']
    acceleration_x = df['AccX [mg]']
    acceleration_y = df['AccY [mg]']
    acceleration_z = df['AccZ [mg]']

    if directory == 'input':
        gyro_x = df['GyroX [mdps]']
        gyro_y = df['GyroY [mdps]']
        gyro_z = df['GyroZ [mdps]']

    # Reste du code pour tracer les graphiques...

def load_csv_file():
    """ Charge un fichier CSV et effectue des analyses et des visualisations. """
    input_folder = simpledialog.askstring("Folder", "Enter the folder of the csv file:")
    directory = verify_folder_input(input_folder)

    if directory:
        csv_file = simpledialog.askstring("CSV File", "Enter the name of the csv file:")
        csv_flag, csv_path = verify_csv_input(csv_file, directory)

        if csv_flag:
            df = pd.read_csv(csv_path)
            if 'T [ms]' not in df.columns:
                add_t_column(df)

            plot_data(df, directory)
        else:
            messagebox.showerror("Error", "Could not find csv file")
    else:
        messagebox.showerror("Error", "Could not find the folder")

# Création de la fenêtre principale
root = tk.Tk()
root.title("Data Analysis Tool")

# Bouton pour charger le fichier CSV
load_button = tk.Button(root, text="Load CSV File", command=load_csv_file)
load_button.pack(padx=10, pady=10)

root.mainloop()
