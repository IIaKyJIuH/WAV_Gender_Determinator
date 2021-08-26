import scipy.stats as stats
import numpy as np

from scipy.io import wavfile

def get_frequencies(wav_file_path: str):
    rate, data = wavfile.read(wav_file_path)

    # получаем доминирующие частоты каждые 200 мс
    step = rate // 5 # 1600 точек каждые 1/5 сек
    window_frequencies = []
    for i in range(0, len(data), step):
        ft = np.fft.fft(data[i:i+step]) # получаем коэффициенты
        freqs = np.fft.fftfreq(len(ft)) # получаем частоты по коэффициентам
        imax = np.argmax(np.abs(ft))
        freq = freqs[imax]
        freq_in_hz = abs(freq * rate)
        window_frequencies.append(freq_in_hz)
    # отфильтруем частоты на "человеческие"
    filtered_frequencies = [f for f in window_frequencies if 75 <= f and f <= 1100]
    return filtered_frequencies

def get_features(frequencies):
    nobs, minmax, mean, variance, skew, kurtosis = stats.describe(frequencies)
    median    = np.median(frequencies)
    mode      = stats.mode(frequencies).mode[0]
    std       = np.std(frequencies)
    low,peak  = minmax
    q75,q25   = np.percentile(frequencies, [75,25])
    iqr       = q75 - q25
    return nobs, mean, skew, kurtosis, median, mode, std, low, peak, q25, q75, iqr
