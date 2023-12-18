import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Lecture des données depuis le fichier CSV
df = pd.read_csv('datas.csv')

# Analyse temporelle : tracé des signaux d'accélération
plt.figure(figsize=(12, 4))
plt.subplot(3, 1, 1)
plt.plot(df['T [ms]'], df['AccX [mg]'], label='AccX')
plt.title('Accélération en X')
plt.subplot(3, 1, 2)
plt.plot(df['T [ms]'], df['AccY [mg]'], label='AccY')
plt.title('Accélération en Y')
plt.subplot(3, 1, 3)
plt.plot(df['T [ms]'], df['AccZ [mg]'], label='AccZ')
plt.title('Accélération en Z')
plt.xlabel('Time (ms)')
plt.ylabel('Acceleration (mg)')
plt.legend()
plt.show()

def fonction_add(a,b):
    

# Préparation pour l'analyse fréquentielle
N = len(df)
T = (df['T [ms]'].iloc[-1] - df['T [ms]'].iloc[0]) / 1000.0  # Durée totale en secondes
f = np.linspace(0, 1 / T, N)

# Fonction pour calculer et tracer la FFT d'un signal d'accélération
def plot_fft(signal, title):
    signal_fft = fft(signal)
    plt.plot(f[:N // 2], 2.0 / N * np.abs(signal_fft[:N // 2]), label=title)

# Analyse fréquentielle : tracé des FFT pour les signaux d'accélération
plt.figure(figsize=(12, 4))
plot_fft(df['AccX [mg]'], 'FFT AccX')
plot_fft(df['AccY [mg]'], 'FFT AccY')
plot_fft(df['AccZ [mg]'], 'FFT AccZ')
plt.title('Analyse Fréquentielle des Signaux d\'Accélération')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Extraction des caractéristiques pour l'entraînement du réseau de neurones
# Ici, vous pouvez extraire des caractéristiques spécifiques comme les fréquences dominantes
# Exemple : df_features = extract_features(df)
# ...

# Enregistrement des caractéristiques dans un nouveau fichier CSV
# df_features.to_csv('features_for_neural_network.csv')
