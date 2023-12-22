import pandas as pd
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
from scipy.fft import fft
import numpy as np
donnees = pd.read_csv('SensorTile_Log_N000.csv')

def butter_lowpass_filter(donnees, cutoff_frequency, sampling_rate, order=1):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_donnees = filtfilt(b, a, donnees)
    return filtered_donnees, b, a

if __name__ == '__main__':
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
