import concurrent.futures
from scipy.io import wavfile
import numpy as np

def STE(frame):
    """ Hàm tính STE của một frame """
    return np.sum(frame**2)

def calculate(signal, frame_shift_amount_in_sample, frame_length_in_sample):
    """ Hàm tính STE của tín hiệu """
    for i in range(0, len(signal), frame_shift_amount_in_sample):
        frame = signal[i:i + frame_length_in_sample]
        if len(frame) == frame_length_in_sample:
            yield STE(frame)

def bSearch(f, g):
    """ Hàm tìm ngưỡng bằng cách dùng Binary Search """
    T = 0
    i = np.sum([1 for ste in f if ste < T])
    p = np.sum([1 for ste in g if ste > T])
    j = q = -1
    T_max, T_min = 1, min(f.min(), g.min())
    while i != j or p != q:
        T = (T_max + T_min) / 2
        left_hand_side = 1 / len(f) * np.sum([(ste - T) for ste in f if ste > T]) \
                        - 1 / len(g) * np.sum([(T - ste) for ste in g if ste < T])
        T_min, T_max = (T, T_max) if left_hand_side > 0 else (T_min, T)
        j, q = i, p
        i = np.sum([1 for ste in f if ste < T])
        p = np.sum([1 for ste in g if ste > T])
    return T

def calculateThread(signal_name):
    """ Tìm ngưỡng STE của tín hiệu được đưa vào và lưu vào list data """
    print(f'Calculating {signal_name}')

    Fs, signal = wavfile.read(f'{path}/{signal_name}.wav')
    signal = signal / np.max((np.max(signal), abs(np.min(signal))))
    frame_shift_amount_in_sample = int(frame_shift_amount_in_second * Fs)
    frame_length_in_sample = int(frame_length_in_second * Fs)
    STE_speech, STE_silent = np.zeros(0), np.zeros(0)

    with open(f'{path}/{signal_name}.lab', 'r') as file_lab:
        lines = file_lab.readlines()[:-2]
        for line in lines:
            start, end, stype = line.split()
            ste = np.array(list(calculate(signal[int(float(start) * Fs):int(float(end) * Fs)]
                                 , frame_shift_amount_in_sample
                                 , frame_length_in_sample)))
            if stype == 'sil':
                STE_silent = np.concatenate((STE_silent, ste))
            else:
                STE_speech = np.concatenate((STE_speech, ste))
        mx = max(STE_speech.max(), STE_silent.max())
        STE_speech /= mx
        STE_silent /= mx
        T = bSearch(STE_speech, STE_silent)
        datas.append((signal_name, T))

    print(f'Calculate {signal_name} done !')

def showResult():
    """ In ra kết quả sau khi tính toán """
    rs = ()
    print('{:^13}{:^13}'
          .format('signal', 'Threshold'))
    for data in datas:
        print('{:^13}{:^13.3f}'.format(*data))
        rs += (data[-1],)
    print(f'Threshold = {np.mean(rs)}')

if __name__ == '__main__':
    # Khởi tạo các thông số mặc định
    datas = []
    frame_shift_amount_in_second, frame_length_in_second = 0.01, 0.024
    path = '../TinHieuHuanLuyen-44k'
    list_of_signals = ['01MDA', '02FVA', '03MAB', '06FTB', '30FTN', '42FQT', '44MTT', '45MDV']  #
    # Duyệt qua từng tín hiệu và thực hiện tính toán
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        future = executor.map(calculateThread, list_of_signals)
        for i in future:
            if i:
                print(i)
    showResult()