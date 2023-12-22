import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft

# Variables globales
data = None
model = None
canvas = None
plot_frame = None

def init():
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_csv_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Fichier chargé avec succès.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def filter_data():
    global data
    if data is not None:
        t_min = simpledialog.askinteger("Time Min", "Enter minimum time (T min in ms):", parent=root)
        t_max = simpledialog.askinteger("Time Max", "Enter maximum time (T max in ms):", parent=root)
        if t_min is not None and t_max is not None:
            data = data[(data['T [ms]'] >= t_min) & (data['T [ms]'] <= t_max)].reset_index(drop=True)

def get_next_filename():
    max_version = 0
    pattern = r'dataset_v(\d+)\.csv'
    directory = 'datasets/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in os.listdir(directory):
        match = re.search(pattern, filename)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)

    return f'{directory}dataset_v{max_version + 1}.csv'

def export_data():
    global data_to_export
    if data_to_export is None or data_to_export.empty:
        messagebox.showinfo("Information", "No data to export.")
        return

    column_value = simpledialog.askinteger("Input", 
                                           "Enter a value for the new column:\n" +
                                           "0: random movements\n" +
                                           "1: Balance movements",
                                           parent=root)
    if column_value is not None:
        data_to_export['State'] = column_value
        columns_to_export = ['AccX [mg]', 'AccY [mg]', 'AccZ [mg]', 'State']
        data_to_export = data_to_export[columns_to_export]

        saved_filename = get_next_filename()
        data_to_export.to_csv(saved_filename, index=False, header=True)
        messagebox.showinfo("Information", f"Data exported to file '{saved_filename}'.")


def apply_filter():
    global data
    if data is not None:
        fs = 1000  # Fréquence d'échantillonnage
        fc = 10    # Fréquence de coupure
        order = 2  # Ordonnée du filtre
        b, a = butter(order, fc / (0.5 * fs), btype='low')
        data['Filtered AccX [mg]'] = filtfilt(b, a, data['AccX [mg]'])

def plot_data():
    global data, canvas, plot_frame
    if data is not None:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(data['T [ms]'], data['AccX [mg]'], label='AccX (mg) en fonction du temps (T)')
        if 'Filtered AccX [mg]' in data.columns:
            ax.plot(data['T [ms]'], data['Filtered AccX [mg]'], label='AccX filtré (mg) en fonction du temps (T)')
        ax.set_xlabel('Temps (ms)')
        ax.set_ylabel('Accélération (mg)')
        ax.set_title('Graphique des données d\'accélération en fonction du temps')
        ax.legend()
        ax.grid(True)

        if canvas:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

def build_model():
    global model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(10,)),  # Ajustez selon la forme de vos données
        Dense(128, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

def save_model():
    global model
    if model:
        file_path = filedialog.asksaveasfilename(title="Enregistrer le modèle", defaultextension=".h5", filetypes=[("H5 Files", "*.h5")])
        if file_path:
            try:
                model.save(file_path)
                messagebox.showinfo("Succès", "Modèle enregistré avec succès.")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'enregistrement du modèle: {e}")

def train_test_split():
    # Ajoutez ici votre logique de séparation des données
    return null

def compile_model():
    global model
    if model:
        opt = SGD(learning_rate=0.001, momentum=0.2)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

def train_model():
    global model, data
    if model and data is not None:
        # Assurez-vous que X_train, y_train sont définis correctement
        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)
        y_train_encoded = LabelEncoder().fit_transform(y_train)
        y_test_encoded = LabelEncoder().fit_transform(y_test)

        callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        history = model.fit(X_train, y_train_encoded, validation_split=0.15, epochs=500, batch_size=30, callbacks=[callback], use_multiprocessing=True)
        print_accuracy(history)

def print_accuracy(history):
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
    global model, data
    if model and data is not None:
        X_test = data.iloc[:, :-1]  # Assurez-vous que ceci correspond à votre structure de données
        y_test = data.iloc[:, -1]
        y_test_encoded = LabelEncoder().fit_transform(y_test)
        scores = model.evaluate(X_test, y_test_encoded)
        print("\nÉvaluation sur le test data %s: %.2f - %s: %.2f%% " % (model.metrics_names[0], scores[0], model.metrics_names[1], scores[1] * 100))

def analyse_frequentiel():
    global data
    if data is not None:
        # Paramètres pour l'analyse fréquentielle
        Te = 1.0 / 1000  # Période d'échantillonnage en secondes
        N = len(data)
        T = N * Te

        # Fréquences pour l'analyse FFT
        f = np.linspace(0.0, 1.0/(2.0*Te), N//2)

        # Calcul de la FFT sur les données filtrées
        yf = fft(data['Filtered AccX [mg]'])
        yf = 2.0/N * np.abs(yf[0:N//2])

        plt.figure(figsize=(10, 6))
        plt.plot(f, yf)
        plt.title('Analyse Fréquentielle de AccX filtré')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Amplitude')
        plt.show()

# Code Tkinter pour l'interface graphique
def main():
    global plot_frame, root
    root = tk.Tk()
    root.title("Interface de Traitement de Données")

    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Boutons
    load_button = tk.Button(button_frame, text="Charger CSV", command=load_csv_file)
    load_button.pack(pady=5)

    filter_button = tk.Button(button_frame, text="Filtrer Données", command=filter_data)
    filter_button.pack(pady=5)

    apply_filter_button = tk.Button(button_frame, text="Appliquer Filtre", command=apply_filter)
    apply_filter_button.pack(pady=5)

    plot_button = tk.Button(button_frame, text="Afficher Graphique", command=plot_data)
    plot_button.pack(pady=5)

    build_model_button = tk.Button(button_frame, text="Construire Modèle", command=build_model)
    build_model_button.pack(pady=5)

    save_model_button = tk.Button(button_frame, text="Enregistrer Modèle", command=save_model)
    save_model_button.pack(pady=5)

    train_button = tk.Button(button_frame, text="Entraîner Modèle", command=train_model)
    train_button.pack(pady=5)

    evaluate_button = tk.Button(button_frame, text="Évaluer Modèle", command=evaluate_model)
    evaluate_button.pack(pady=5)

    freq_analysis_button = tk.Button(button_frame, text="Analyse Fréquentielle", command=analyse_frequentiel)
    freq_analysis_button.pack(pady=5)

    root.mainloop()

