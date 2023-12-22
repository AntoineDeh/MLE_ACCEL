import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pathlib

def strike(text):
    return ''.join([c + '\u0336' for c in text])

def print_dataset(dataset, frame, title='Dataset state'):
    for widget in frame.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(dataset['AccX [mg]'], label='AccX as a function of T')
    axs[0].plot(dataset['AccY [mg]'], label='AccY as a function of T')
    axs[0].plot(dataset['AccZ [mg]'], label='AccZ as a function of T')
    axs[0].set_xlabel('Point number')
    axs[0].set_ylabel('Acceleration')
    axs[0].set_title(title)
    axs[0].legend()

    axs[1].plot(dataset['State'])
    axs[1].set_xlabel('Point number')
    axs[1].set_ylabel('State')
    axs[1].grid(True)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def add_file(dataset, headers, graph_frame):
    file_path = filedialog.askopenfilename(initialdir="./output", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path, usecols=headers)
        if data.shape[0] % 100 != 0:
            messagebox.showerror("Error", f"File {file_path} has {data.shape[0]} rows, which is not a multiple of 100.")
            return

        add_data = messagebox.askyesno("Confirm", "Do you want to add these values to the dataset?")
        if add_data:
            dataset = pd.concat([dataset, data], ignore_index=True)
            print_dataset(dataset, graph_frame, 'Actual dataset')
    return dataset

def export_dataset(dataset):
    dataset_name = simpledialog.askstring("Dataset Name", "Enter the name of the dataset to export:")
    if dataset_name:
        dataset.to_csv(f'datasets/{dataset_name}.csv', index=False)
        messagebox.showinfo("Success", f"Dataset exported as 'datasets/{dataset_name}.csv'.")

# Création de la fenêtre principale
root = tk.Tk()
root.title("Dataset Manager")

headers = ["AccX [mg]", "AccY [mg]", "AccZ [mg]", "State"]
dataset = pd.DataFrame()

button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, padx=10, pady=10)

add_button = tk.Button(button_frame, text="Add File", command=lambda: add_file(dataset, headers, graph_frame))
add_button.pack(side=tk.LEFT, padx=10)

export_button = tk.Button(button_frame, text="Export Dataset", command=lambda: export_dataset(dataset))
export_button.pack(side=tk.LEFT, padx=10)

graph_frame = tk.Frame(root)
graph_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
