import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result


def print_dataset(dataset, title: str = 'Dataset state'):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(dataset['AccX [mg]'], label='AccX as a function of T')
    plt.plot(dataset['AccY [mg]'], label='AccY as a function of T')
    plt.plot(dataset['AccZ [mg]'], label='AccZ as a function of T')
    plt.xlabel('Point number')
    plt.ylabel('Acceleration')
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(dataset['State'])
    plt.xlabel('Point number')
    plt.ylabel('State')

    plt.grid(True)
    plt.show()


files_path = [f for f in pathlib.Path().glob("./output/*.csv")]
for index, file_path in enumerate(files_path):
    print(f"{index} - {file_path}")

headers = ["AccX [mg]", "AccY [mg]", "AccZ [mg]", "State"]
dataset = pd.DataFrame()

while True:
    entry = input("Enter the id of the file you want to add to the dataset or 'exit' to exit:")

    if entry.lower() == 'exit' or entry.lower() == '':
        break

    file_selected = files_path[int(entry)]

    data = pd.read_csv(file_selected, usecols=headers)
    if data.shape[0] % 100:
        print(f"File {file_selected} has {data.shape[0]} rows, which is not a multiple of 100, so it will be ignored")
        continue
    else:
        print(f"File selected: {file_selected.name}")

    action = input("Do you want to add this values in the dataset (Y/n): ")
    if action.lower() == 'y' or action.lower() == 'yes' or action.lower() == '':
        dataset = pd.concat([dataset, data], ignore_index=True)
        print_dataset(dataset, 'Actual dataset')

print_dataset(dataset)

dataset_name = input("Enter the name of the dataset to export: ")
dataset.to_csv(f'datasets/{dataset_name}.csv', index=False, header=True)
