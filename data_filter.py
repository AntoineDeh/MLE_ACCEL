import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

FILTRE = 0

if __name__ == '__main__':
    # Lire les données à partir du fichier CSV
    donnees = pd.read_csv('input/SansBalancier.csv')
    # donnees = donnees[(donnees['T [ms]'] >= 12500) & (donnees['T [ms]'] <= 27500)]
    # donnees.reset_index(drop=True, inplace=True)

    if FILTRE:
        # Fréquence d'échantillonnage
        fs = 1 / ((donnees['T [ms]'][1] - donnees['T [ms]'][0]) / 1000.0)  # Conversion de ms en secondes

        # Fréquence de coupure du filtre passe-bas (en Hz)
        fc = 10  # Exemple de fréquence de coupure, à ajuster selon les besoins

        # Ordonnée du filtre
        ordre = 2  # L'ordre du filtre Butterworth

        # Créer le filtre Butterworth
        b, a = butter(ordre, fc / (0.5 * fs), btype='low', analog=False)

        # Appliquer le filtre aux données
        donnees_acc_y_filtrees = filtfilt(b, a, donnees['AccY [mg]'])


    # Créer un graphique
    plt.figure(figsize=(10, 6))
    plt.plot(donnees['T [ms]'], donnees['AccX [mg]'], label='AccX en fonction de T')
    if (FILTRE):
        plt.plot(donnees['T [ms]'], donnees_acc_y_filtrees, label='AccY butter en fonction de T')
    plt.plot(donnees['T [ms]'], donnees['AccY [mg]'], label='AccY en fonction de T')
    plt.plot(donnees['T [ms]'], donnees['AccZ [mg]'], label='AccZ en fonction de T')
    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Accélération')
    plt.title('Données de l\'accéléromètre en fonction de T')
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 6))

    plt.plot(donnees['T [ms]'], donnees['GyroY [mdps]'], label='GyroY en fonction de T')
    plt.plot(donnees['T [ms]'], donnees['GyroX [mdps]'], label='GyroX en fonction de T')
    plt.plot(donnees['T [ms]'], donnees['GyroZ [mdps]'], label='GyroZ en fonction de T')

    plt.xlabel('Temps (T) en ms')
    plt.ylabel('Gyroscope')
    plt.title('Données du gyroscope en fonction de T')
    plt.legend()
    plt.grid(True)
    plt.show()