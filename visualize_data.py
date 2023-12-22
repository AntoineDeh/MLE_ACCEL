import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def verify_folder_input(user_input_folder):
    folder_authorized = False
    output = None
    if user_input_folder == 'input' or user_input_folder == 'in' or user_input_folder == 'i':
        output = "input"
        folder_authorized = True
    elif user_input_folder == 'output' or user_input_folder == 'out' or user_input_folder == 'o':
        output = "output"
        folder_authorized = True
    elif (user_input_folder == 'dataset' or user_input_folder == 'data'
          or user_input_folder == 'datasets' or user_input_folder == 'd'):
        output = "datasets"
        folder_authorized = True

    return folder_authorized, output


def verify_csv_input(user_input_csv, checked_directory):
    test_path = os.path.join(checked_directory, user_input_csv + '.csv')
    return os.path.isfile(test_path), test_path


# Create Time columns for visualization
def add_t_column(csv_df):
    # Create an incremental 'T [ms]' column with a step size of 20
    csv_df['T [ms]'] = range(0, 20 * len(df), 20)
    # Display the DataFrame with the new 'T [ms]' column
    # print(df)
    # You can use df.plot() or any other visualization library to plot the data


input_folder = input("Enter the folder of the csv file :")
folder_flag, directory = verify_folder_input(input_folder)
if folder_flag:
    csv_file = input("Enter the name of the csv file :")
    csv_flag, csv_path = verify_csv_input(csv_file, directory)
    if csv_flag:

        # Charger le fichier CSV
        df = pd.read_csv(csv_path)
        # Verification de la présence de la colonne temps ou création temporaire
        try:
            temps = df['T [ms]']
        except:
            add_t_column(df)
            temps = df['T [ms]']

        # Attribution des colonnes du csv vers dans des variables
        acceleration_x = df['AccX [mg]']
        acceleration_y = df['AccY [mg]']
        acceleration_z = df['AccZ [mg]']

        if directory == 'input':
            gyro_x = df['GyroX [mdps]']
            gyro_y = df['GyroY [mdps]']
            gyro_z = df['GyroZ [mdps]']

        data = df['AccX [mg]'].values

        # Calcul de la DFT
        dft_result = np.fft.fft(data)

        # Calcul de la FFT
        fft_result = np.fft.fft(data)

        # Fréquences associées à la transformation
        freq = np.fft.fftfreq(len(data))

        # Tracé des résultats
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(freq, np.abs(dft_result))
        plt.title('Transformée de Fourier Discrète (DFT)')
        plt.xlabel('Fréquence')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(freq, np.abs(fft_result))
        plt.title('Transformée de Fourier Rapide (FFT)')
        plt.xlabel('Fréquence')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

        # Plot des données
        plt.figure(figsize=(12, 6))
        plt.plot(temps, acceleration_x, label='Accélération X')
        plt.plot(temps, acceleration_y, label='Accélération Y')
        plt.plot(temps, acceleration_z, label='Accélération Z')

        plt.title('Données de l\'accéléromètre au fil du temps')
        plt.xlabel('Temps')
        plt.ylabel('Accélération')
        plt.legend()
        plt.show()

        # Plot en 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(temps, acceleration_x, acceleration_y, label='Trajectoire 3D de l\'accélération')
        ax.set_xlabel('Temps')
        ax.set_ylabel('Accélération X')
        ax.set_zlabel('Accélération Y')
        ax.legend()
        plt.show()

        # Plot des données
        if directory == 'input':
            plt.figure(figsize=(12, 6))
            plt.plot(temps, gyro_x, label='Gyro X')
            plt.plot(temps, gyro_y, label='Gyro Y')
            plt.plot(temps, gyro_z, label='Gyro Z')

            plt.title('Données du gyroscope au fil du temps')
            plt.xlabel('Temps')
            plt.ylabel('Gyro')
            plt.legend()
            plt.show()

            # Plot en 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(temps, gyro_x, gyro_y, label='Trajectoire 3D du gyroscope')
            ax.set_xlabel('Temps')
            ax.set_ylabel('Gyro X')
            ax.set_zlabel('Gyro Y')
            ax.legend()
            plt.show()

        # Plot Movement Points (if output folder)
        if directory == 'output' or directory == 'datasets':
            movement_indices = df[df['State'] == 1].index
            plt.figure(figsize=(12, 6))
            plt.plot(temps, acceleration_x, label='AccX')
            plt.plot(temps, acceleration_y, label='AccY')
            plt.plot(temps, acceleration_z, label='AccZ')
            plt.scatter(temps[movement_indices], acceleration_x[movement_indices], color='red', label='Movement Points (AccX)')
            plt.scatter(temps[movement_indices], acceleration_y[movement_indices], color='red', label='Movement Points (AccY)')
            plt.scatter(temps[movement_indices], acceleration_z[movement_indices], color='red', label='Movement Points (AccZ)')

            plt.title('Acceleration Data with Movement Points')
            plt.xlabel('Time')
            plt.ylabel('Acceleration')
            plt.legend()
            plt.show()

    else:
        print("Could not find csv file")
else:
    print("Could not find the folder")
