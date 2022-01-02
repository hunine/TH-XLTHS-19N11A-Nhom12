import concurrent.futures
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def find_dips(l):
    return np.array([i for i in range(1, l.size - 1) if l[i - 1] > l[i] and l[i + 1] > l[i] and l[i] <= amdf_threshold]) if l.size > 2 else np.zeros(0)

def amdf(frame, max_period_in_sample, min_period_in_sample):
    d = np.zeros(max_period_in_sample + 1)
    if frame.size >= max_period_in_sample:
        for n in range(min_period_in_sample, max_period_in_sample + 1):
            d[n] = np.sum(abs(frame[:frame.size - n] - frame[n:]) / (frame.size - n))
    else:
        return -1

    d /= np.max(d)

    dips = find_dips(d)

    if dips.size == 0:
        return -1

    # plt.plot(d)
    # plt.plot([0, d.size], [amdf_threshold, amdf_threshold])
    # plt.plot(dips[np.argmin(d[dips])], d[dips[np.argmin(d[dips])]], 'ro')
    # plt.show()
    # plt.pause(1)block=False
    # plt.close()

    return dips[np.argmin(d[dips])]

def calculateThread(signal_name):
    print(f'Calculating {signal_name}')

    Fs, signal = wavfile.read(f'{path}{signal_name}.wav')
    signal = signal / max(np.max(signal), abs(np.min(signal)))

    with open(f'{path}{signal_name}.lab', 'r') as file_lab:
        F0_mean_lab = float(file_lab.readlines()[-2].split()[1])

    frame_length_in_sample = int(frame_length_in_seconds * Fs)
    max_period_in_sample, min_period_in_sample = Fs // min_pitch_value, Fs // max_pitch_value
    pitchs = np.zeros(0)

    for i in range(0, signal.size, frame_length_in_sample):
        T0_in_sample = amdf(signal[i:i + frame_length_in_sample], max_period_in_sample, min_period_in_sample)

        if T0_in_sample != -1:
            pitchs = np.append(pitchs, [Fs / T0_in_sample])

    pitchs = medfil(pitchs)

    datas.append((signal_name, np.round(np.mean(pitchs)), F0_mean_lab))

    print(f'Calculate {signal_name} done !')

def medfil(l):
    l = np.concatenate(([0], l, [0]))
    indexes = []

    for i in range(1, l.size - 1):
        if abs(l[i] - l[i - 1]) <= pitch_shift and abs(l[i] - l[i + 1]) <= pitch_shift:
            indexes.append(i)

    l = l[indexes]

    for i in range(2, l.size - 2):
        l[i] = np.mean(l[i - 2:i + 2])

    return l

def genderDetect(name, F0mean):
    if 85 <= F0mean <= 155:
        return 'Male', 100 if name[2] == 'M' else 0
    if 155 < F0mean <= 255:
        return 'Female', 100 if name[2] == 'F' else 0
    return 'Unknown', 0

if __name__ == '__main__':
    list_of_signals = ['01MDA', '02FVA', '03MAB', '06FTB', '30FTN', '42FQT', '44MTT', '45MDV'] #
    path = '../TinHieuHuanLuyen-44k/'
    amdf_threshold = 0.528
    frame_length_in_seconds = 0.025
    max_pitch_value = 400
    min_pitch_value = 75
    pitch_shift = 10
    datas = []

    with concurrent.futures.ThreadPoolExecutor(20) as executor:
        future = executor.map(calculateThread, list_of_signals)
        for i in future:
            if i:
                print(i)

    print(f'{"File":^10}{"AMDF (Hz)":^10}{"Lab (Hz)":^10}Relative Error (%) {"Gender":^10} Accuracy (%)')

    for name, F0mean_amdf, F0mean_lab in datas:

        gender, accuracy = genderDetect(name, F0mean_amdf)

        print(f'{name:^10}{F0mean_amdf:^10.1f}{F0mean_lab:^10.1f}{abs(F0mean_amdf - F0mean_lab) / F0mean_lab * 100:^19.1f}{gender:^10}{accuracy:^13}')