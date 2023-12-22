import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Initialisation des variables globales
canvas = None
plot_frame = None
model = None
X_train, X_test, y_train, y_test = None, None, None, None
y_train_encoded, y_test_encoded = None, None
history = None
data = None
root = None

def init():
    # Initialisation de la graine aléatoire pour la reproductibilité
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_csv_file():
    """ Charge un fichier CSV et retourne les données. """
    global root
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return None
    else:
        return None

def filter_data(data):
    """ Demande à l'utilisateur s'il souhaite filtrer les données et les filtre si nécessaire. """
    global root
    t_min = simpledialog.askinteger("Time Min", "Enter minimum time (T min in ms):", parent=root)
    t_max = simpledialog.askinteger("Time Max", "Enter maximum time (T max in ms):", parent=root)
    if t_min is not None and t_max is not None:
        filtered_data = data[(data['T [ms]'] >= t_min) & (data['T [ms]'] <= t_max)]
        return filtered_data.reset_index(drop=True)
    return data

def apply_filter(data, fs, fc, order):
    """ Applique un filtre passe-bas Butterworth aux données. """
    b, a = butter(order, fc / (0.5 * fs), btype='low', analog=False)
    return filtfilt(b, a, data)

def plot_data(data):
    """ Effectue des visualisations des données. """
    global canvas, plot_frame
    if data.empty:
        messagebox.showinfo("Info", "No data to plot.")
        return

    fs = 1 / ((data['T [ms]'][1] - data['T [ms]'][0]) / 1000.0)  # Fréquence d'échantillonnage
    fc = 10  # Fréquence de coupure
    order = 2  # Ordonnée du filtre

    data_acc_y_filtered = apply_filter(data['AccY [mg]'], fs, fc, order)

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(data['T [ms]'], data['AccX [mg]'], label='AccX (mg) en fonction du temps (T)')
    ax.plot(data['T [ms]'], data_acc_y_filtered, label='AccY filtré (mg) en fonction du temps (T)')
    ax.plot(data['T [ms]'], data['AccZ [mg]'], label='AccZ (mg) en fonction du temps (T)')
    ax.set_xlabel('Temps (ms)')
    ax.set_ylabel('Accélération (mg)')
    ax.set_title('Graphique des données d\'accélération en fonction du temps')
    ax.legend()
    ax.grid(True)

    if canvas is not None:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def build_model():
    """ Construction du modèle """
    global model, X_train
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

def save_model(model):
    """ Enregistre le modèle entraîné """
    file_path = filedialog.asksaveasfilename(title="Enregistrer le modèle", defaultextension=".h5",
                                             filetypes=[("H5 Files", "*.h5")])
    if file_path:
        try:
            model.save(file_path)
            messagebox.showinfo("Succès", "Modèle enregistré avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'enregistrement du modèle: {e}")

def train_test_split_data():
    """ Sépare les données en ensembles d'entraînement et de test """
    global X_train, X_test, y_train, y_test, data
    # Ici, vous devez définir x et y en fonction de vos données
    # Exemple : x = data.drop('label_column', axis=1), y = data['label_column']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def compile_model():
    """ Compilation du modèle """
    global model
    opt = SGD(learning_rate=0.001, momentum=0.2)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

def train_model():
    """ Entraînement du modèle """
    global model, X_train, y_train, history
    callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # Ici, assurez-vous que y_train est encodé si nécessaire
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=500, batch_size=30, callbacks=[callback], use_multiprocessing=True)

def print_accuracy():
    """ Visualisation de la perte et de la précision """
    global history
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

def evaluate_model():
    """ Évaluation du modèle """
    global model, X_test, y_test
    scores = model.evaluate(X_test, y_test)
    print("\nÉvaluation sur le test data %s: %.2f - %s: %.2f%% " % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

def main_menu():
    global plot_frame, root
    root = tk.Tk()
    root.title("Data Analysis and Model Training")

    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def load_and_plot():
        global data
        data = load_csv_file()
        if data is not None:
            plot_data(data)

    def filter_and_plot():
        global data
        data = filter_data(data)
        plot_data(data)

    def build_save_and_train_model():
        global model
        build_model()
        save_model(model)
        train_test_split_data()
        compile_model()
        train_model()
        print_accuracy()

    def evaluate():
        evaluate_model()

    # Menu buttons
    tk.Button(root, text='Load CSV and Plot', command=load_and_plot).pack(fill=tk.X)
    tk.Button(root, text='Filter Data and Replot', command=filter_and_plot).pack(fill=tk.X)
    tk.Button(root, text='Build, Save and Train Model', command=build_save_and_train_model).pack(fill=tk.X)
    tk.Button(root, text='Evaluate Model', command=evaluate).pack(fill=tk.X)

    root.mainloop()

if __name__ == "__main__":
    init()
    main_menu()



'''

if __name__ == "__main__":
   init()
    load_csv_file()
    
   #Pour tous les fichiers de Datas faire :
    filter_data()
    apply_filter()
    plot_data()
    analyse_frequentiel() // avoir un bouton suivant ensuite pour passer au suivant // chacun de ces fichiers vont dans le dossier output
    ############""puis 

    build_model()
    save_model()

    train_test()
    compile()


    train()
    print_accuracy()
  ///Autant de fois que l'on veut on peut faire : 
    evaluation() // un bouton suivant apparait après pour passer à un autre ou faire exit

'''