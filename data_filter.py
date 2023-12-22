import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

def on_filter_button_click():
    global donnees
    donnees_filtrees = filter_data(donnees)
    plot_data(donnees_filtrees)

def main():
    global donnees
    donnees = load_csv_file()
    if donnees is not None:
        plot_data(donnees)
        filter_button['state'] = 'normal'

# Création de la fenêtre principale
root = tk.Tk()
root.title("Data Analysis Tool")

donnees = None
canvas = None

# Cadre pour le plot
plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Bouton pour filtrer les données
filter_button = tk.Button(root, text="Filtrer", command=on_filter_button_click, state='disabled')
filter_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Démarrage automatique du traitement des données
root.after(100, main)

root.mainloop()
