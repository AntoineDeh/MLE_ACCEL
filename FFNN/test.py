'''
ouvrir un fichier
plot
filtrer
plot
labeliser-pre traitement
on enregistre dans datasets avec les filtres

création de l'entrainement

build model
save model h5 et json

evaluate plot


README

'''

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

import pandas as pd
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np

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



def init():
    # Initialisation de la graine aléatoire pour la reproductibilité
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_csv_file():
    """ Charge un fichier CSV et retourne les données. """
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

    global canvas
    if canvas is not None:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)



def build_model(input_shape1, input_shape2):
   # Construction du modèle
    model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
    ])

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

def train_test():
    # Séparation en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

#model.summary()

def compile():
    # Compilation du modèle
    opt = SGD(learning_rate=0.001, momentum=0.2)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

def train():
    # Entraînement avec validation et arrêt anticipé
    callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(X_train, y_train_encoded, validation_split=0.15, epochs=500, batch_size=30, callbacks=[callback], use_multiprocessing=True)

def print_accuracy():
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

def evaluation():
    # Évaluation du modèle
    scores = model.evaluate(X_test, y_test_encoded)
    print("\nÉvaluation sur le test data %s: %.2f - %s: %.2f%% " % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))


def analyse_frequentiel(donnees):
    tau = 0.01
    Te = tau / 1000
    Fe = 1 / Te
    NFFT = 2**20
    f = np.arange(NFFT // 2 + 1) / NFFT
    F = f * Fe

    cutoff_frequency = 0.25  # Exemple de fréquence de coupure, à ajuster selon les besoins
    order = 1  # Ordre du filtre Butterworth

    num, den = butter(order, 2 * cutoff_frequency)
    H_PB = freqz(num, den, worN=f, fs=1)

    # Affichage dans le domaine fréquentiel
    signal_x = donnees['AccX [mg]']
    signal_y = donnees['AccY [mg]']
    signal_z = donnees['AccZ [mg]']
    fft_result_x = fft(signal_x)
    fft_result_y= fft(signal_y)
    fft_result_z = fft(signal_z)
    freq = np.fft.fftfreq(len(signal_x), d=(donnees['T [ms]'][1] - donnees['T [ms]'][0]) / 1000.0)

    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(fft_result_x), label='AccX [mg]')
    plt.plot(freq, np.abs(fft_result_z), label='AccY [mg]')
    plt.plot(freq, np.abs(fft_result_z), label='AccZ [mg]')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Domaine fréquentiel des données de l\'accéléromètre (AccX , AccYet AccZ)')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    init()
    load_csv_file()
    
    filter_data()
    apply_filter()
    plot_data()

    build_model()
    save_model()

    train_test()
    compile()

    train()
    print_accuracy()
    evaluation()
    analyse_temporelle()
    