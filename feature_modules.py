import numpy as np
from statsmodels.tsa.ar_model import AR
from scipy.fftpack import dct, dst
from pywt import wavedec
from scipy.signal import butter, lfilter, freqz, welch
from scipy.integrate import simps

# Statistical Features
def stat_mean(signal, n_channel=19):
    output = []
    for nch in range(n_channel):
        output.append(np.mean(signal, axis=0)[nch])
    return output


def stat_variance(signal, n_channel=19):
    output = []
    for nch in range(n_channel):
        output.append(np.var(signal, axis=0)[nch])
    return output


def stat_correlation(signal, n_channel=19):
    output = np.zeros((n_channel, n_channel))
    for nch1 in range(n_channel):
        for nch2 in range(n_channel):
            output[nch1, nch2] = np.correlate(signal[:, nch1], signal[:, nch2])
    output = list(np.ravel(output))
    return output

# Entropy Features
def shannon_entropy(signal, n_channel = 19):
    output = []
    for nch in range(n_channel):
        eeg_data = signal[:, nch]
        hist, bins = np.histogram(eeg_data, bins=50, density=False)
        hist = hist / np.sum(hist)
        output.append(-np.sum(np.log2(hist[hist > 0]) * hist[hist > 0]))
    return output

def renyi_entropy(signal, alpha_list,n_channel=19):
    output =[]
    for alpha in alpha_list:
        for nch in range(n_channel):
            eeg_data = signal[:, nch]
            hist, bins = np.histogram(eeg_data, bins=50, density=False)
            hist = hist / np.sum(hist)
            output.append(1.0/(1.0 - alpha) * np.log2(np.sum(hist[hist > 0] ** alpha)))
    return output

def tsalis_entropy(signal, q_list, n_channel = 19):
    output = []
    for q in q_list:
        for nch in range(n_channel):
            eeg_data = signal[:, nch]
            hist,bins= np.histogram(eeg_data, bins=50,density=False)
            hist = hist / np.sum(hist)
            output.append(1.0 / (1.0 - q) * (np.sum(hist[hist > 0] ** q) - 1))
    return output

def Approximate_entropy(signal, m, n_channel = 19):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        r = 0.18 * np.std(eeg_data)
        x = [[eeg_data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    output = []
    for nch in range(n_channel):
        eeg_data = signal[:, nch]
        N = len(eeg_data)
        output.append(abs(_phi(m + 1) - _phi(m)))

    return output


def param_ar(signal, max_lag_list, n_channel=19):
    output = []
    for max_lag in max_lag_list:
        for nch in range(n_channel):
            model = AR(signal[:, nch])
            model_fit = model.fit(maxlag=max_lag)
            tmp = model_fit.params
            output = output + list(tmp)
    return output

# Frequency Features
def freq_dct(signal, n_channel=19):
    output = []
    for nch in range(n_channel):
        tmp = dct(signal[:, nch], norm='ortho')
        output = output + list(tmp)
    return output

def freq_dst(signal, n_channel=19):
    output = []
    for nch in range(n_channel):
        tmp = dst(signal[:, nch], norm='ortho')
        output += list(tmp)
    return output

def wavelet(signal, family_list, n_channel=19):
    output = []
    for fml in family_list:
        for nch in range(n_channel):
            wave_coeff = wavedec(signal[:, nch], fml, level=3)
            for tmp in wave_coeff:
                output += list(tmp)
    return output

# Diff features
def mean_dif_channels(signal):
    fp  = signal[:, 1] - signal[:, 2]
    f1  = signal[:, 11] - signal[:, 12]
    f2  = signal[:, 3] - signal[:, 4]
    t   = signal[:, 13] - signal[:, 14]
    c   = signal[:, 5] - signal[:, 6]
    p1  = signal[:, 15] - signal[:, 16]
    p2  = signal[:, 7] - signal[:, 8]
    o   = signal[:, 9] - signal[:, 10]
    return [np.mean(fp), np.mean(f1), np.mean(f2), np.mean(t), np.mean(c), np.mean(p1), np.mean(p2), np.mean(o)]

def var_dif_channels(signal):
    fp  = signal[:, 1] - signal[:, 2]
    f1  = signal[:, 11] - signal[:, 12]
    f2  = signal[:, 3] - signal[:, 4]
    t   =  signal[:, 13] - signal[:, 14]
    c   = signal[:, 5] - signal[:, 6]
    p1  = signal[:, 15] - signal[:, 16]
    p2  = signal[:, 7] - signal[:, 8]
    o   = signal[:, 9] - signal[:, 10]
    return [np.var(fp), np.var(f1), np.var(f2), np.var(t), np.var(c), np.var(p1), np.var(p2), np.var(o)]

def cor_dif_channels(signal):
    fp  = signal[:, 1] - signal[:, 2]
    f1  = signal[:, 11] - signal[:, 12]
    f2  = signal[:, 3] - signal[:, 4]
    t   = signal[:, 13] - signal[:, 14]
    c   = signal[:, 5] - signal[:, 6]
    p1  = signal[:, 15] - signal[:, 16]
    p2  = signal[:, 7] - signal[:, 8]
    o   = signal[:, 9] - signal[:, 10]
    #type -> List[nparray]
    output = [np.correlate(fp, fp), np.correlate(fp, f1), np.correlate(fp, f2), np.correlate(fp, t), np.correlate(fp, c), np.correlate(fp, p1), np.correlate(fp, p2), np.correlate(fp, o)]
    output += [np.correlate(f1, fp), np.correlate(f1, f1), np.correlate(f1, f2), np.correlate(f1, t), np.correlate(f1, c), np.correlate(f1, p1), np.correlate(f1, p2), np.correlate(f1, o)]
    output += [np.correlate(f2, fp), np.correlate(f2, f1), np.correlate(f2, f2), np.correlate(f2, t), np.correlate(f2, c), np.correlate(f2, p1), np.correlate(f2, p2), np.correlate(f2, o)]
    output += [np.correlate(t, fp), np.correlate(t, f1), np.correlate(t, f2), np.correlate(t, t), np.correlate(t, c), np.correlate(t, p1), np.correlate(t, p2), np.correlate(t, o)]
    output += [np.correlate(c, fp), np.correlate(c, f1), np.correlate(c, f2), np.correlate(c, t), np.correlate(c, c), np.correlate(c, p1), np.correlate(c, p2), np.correlate(c, o)]
    output += [np.correlate(p1, fp), np.correlate(p1, f1), np.correlate(p1, f2), np.correlate(p1, t), np.correlate(p1, c), np.correlate(p1, p1), np.correlate(p1, p2), np.correlate(p1, o)]
    output += [np.correlate(p2, fp), np.correlate(p2, f1), np.correlate(p2, f2), np.correlate(p2, t), np.correlate(p2, c), np.correlate(p2, p1), np.correlate(p2, p2), np.correlate(p2, o)]
    output += [np.correlate(o, fp), np.correlate(o, f1), np.correlate(o, f2), np.correlate(o, t), np.correlate(o, c), np.correlate(o, p1), np.correlate(o, p2), np.correlate(o, o)]
    return output


def butter_lowpass(cutoff, fs = 512, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, n_channel = 19, fs = 512, order=5):
    output = []
    for ch in range(n_channel):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data[:, ch])
        output.append(y)

    output = np.transpose(np.array(output))
    return output

def power_bond(data, low_freq, high_freq, fs=512):
    win = 4 * fs
    freqs, psd = welch(data, fs, nperseg=win)
    idx_delta = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    freq_res = freqs[1] - freqs[0]
    delta_power = simps(psd[idx_delta], dx=freq_res)
    return delta_power

def power_bond_set1(data, n_channel=19):
    output = []
    for ch in range(n_channel):
        eeg_data = data[:, ch]
        output.append(power_bond(eeg_data, 0, 3.5))
        output.append(power_bond(eeg_data, 3.5, 7.5))
        output.append(power_bond(eeg_data, 7.5, 13.5))
        output.append(power_bond(eeg_data, 13.5, 20))

    return output


def power_bond_set2(data, n_channel=19):
    output = []
    for ch in range(n_channel):
        eeg_data = data[:, ch]
        output.append(power_bond(eeg_data, 0, 3.5))
        output.append(power_bond(eeg_data, 3.5, 7))
        output.append(power_bond(eeg_data, 7.5, 13))
        output.append(power_bond(eeg_data, 13, 15))
        output.append(power_bond(eeg_data, 15, 17))
        output.append(power_bond(eeg_data, 18, 25))
        output.append(power_bond(eeg_data, 25.5, 30))
    return output


def power_bond_set3(data, n_channel=19):
    output = []
    for ch in range(n_channel):
        eeg_data = data[:, ch]
        output.append(power_bond(eeg_data, 1.5, 6))
        output.append(power_bond(eeg_data, 6, 8))
        output.append(power_bond(eeg_data, 8.5, 10))
        output.append(power_bond(eeg_data, 10.5, 12))
        output.append(power_bond(eeg_data, 12.5, 18))
        output.append(power_bond(eeg_data, 18.5, 21))
        output.append(power_bond(eeg_data, 21, 30))
        output.append(power_bond(eeg_data, 30, 40))
    return output


def power_bond_set4(data, n_channel=19):
    output = []
    for ch in range(n_channel):
        eeg_data = data[:, ch]
        output.append(power_bond(eeg_data, 0, 10))
        output.append(power_bond(eeg_data, 10, 20))
        output.append(power_bond(eeg_data, 20, 30))
        output.append(power_bond(eeg_data, 30, 40))
        output.append(power_bond(eeg_data, 40, 50))
        output.append(power_bond(eeg_data, 50, 60))
        output.append(power_bond(eeg_data, 60, 70))
        output.append(power_bond(eeg_data, 70, 80))
        output.append(power_bond(eeg_data, 80, 90))
        output.append(power_bond(eeg_data, 90, 100))
    return output