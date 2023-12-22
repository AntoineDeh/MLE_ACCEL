import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import re

data = None
data_to_export = None

def load_file():
    global data, file_name
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            file_name = file_path.split('/')[-1].split('.')[0]
            data = pd.read_csv(file_path)
            plot_data(data, 'Accelerometer Data as a Function of T')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def plot_data(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['T [ms]'], data['AccX [mg]'], label='AccX as a function of T')
    ax.plot(data['T [ms]'], data['AccY [mg]'], label='AccY as a function of T')
    ax.plot(data['T [ms]'], data['AccZ [mg]'], label='AccZ as a function of T')
    ax.set_xlabel('Time (T) in ms')
    ax.set_ylabel('Acceleration')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

def filter_data():
    global data_to_export
    t_min = simpledialog.askinteger("Input", "Enter T min (in ms):", parent=root)
    t_max = simpledialog.askinteger("Input", "Enter T max (in ms):", parent=root)
    if t_min is not None and t_max is not None:
        filtered_data = data[(data['T [ms]'] >= t_min) & (data['T [ms]'] <= t_max)]
        data_to_export = filtered_data.copy()
        messagebox.showinfo("Information", f"Number of selected rows: {len(filtered_data)}")
        plot_data(filtered_data, 'Filtered Accelerometer Data')

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

def initial_load():
    load_file()
    if data is not None:
        filter_button['state'] = 'normal'
        export_button['state'] = 'normal'

# Create the main window
root = tk.Tk()
root.title("Data Processing Tool")

# Boutons désactivés au départ
filter_button = tk.Button(root, text="Filter Data", command=filter_data, state='disabled')
filter_button.pack(side=tk.LEFT, padx=10, pady=10)

export_button = tk.Button(root, text="Export Data", command=export_data, state='disabled')
export_button.pack(side=tk.LEFT, padx=10, pady=10)

# Lancement initial du chargement des données
root.after(100, initial_load)

root.mainloop()
