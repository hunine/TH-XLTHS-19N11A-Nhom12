from os.path import join
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH_THHL = 'D:\Documents\XLTHS\BT nhom\TinHieuHuanLuyen'
FILE_WAV_THHL = ['phone_F1.wav', 'phone_M1.wav', 'studio_F1.wav', 'studio_M1.wav']
FILE_LAB_THHL = ['phone_F1.lab', 'phone_M1.lab', 'studio_F1.lab', 'studio_M1.lab']

FILE_PATH_THKT = 'D:\Documents\XLTHS\BT nhom\TinHieuKiemThu'
FILE_WAV_THKT = ['phone_F2.wav', 'phone_M2.wav', 'studio_F2.wav', 'studio_M2.wav']
FILE_LAB_THKT = ['phone_F2.lab', 'phone_M2.lab', 'studio_F2.lab', 'studio_M2.lab']
INDEX = 0
TIME_FRAME = 0.03

def readFileInput(fileName):
    inputFile = []
    with open(join(FILE_PATH_THKT, fileName)) as file:
        for line in file:
            inputFile.append(line.split())
    for i in range(0, len(inputFile) - 2):
        inputFile[i][0] = int(float(inputFile[i][0]) * frequency)
        inputFile[i][1] = int(float(inputFile[i][1]) * frequency)
    return inputFile

def getFramesArray(signal, frameLength):
    step = frameLength // 2
    frames = []
    index = 0
    for i in range(0, len(signal) // step):
        temp = signal[index : index + frameLength]
        frames.append(temp)
        index += step
    return frames

# Hàm tự tương quan
def ACF(frame, frameLength):
    N = len(frame)
    xx = np.zeros(N)
    frame = np.concatenate((frame, np.zeros(N)))
    for n in range(N):
        xx[n] = np.sum([frame[m] * frame[m + n] for m in range(frameLength - n)])
    xx = xx / np.max(xx)
    return xx

# Tìm đỉnh của 1 frame
def getHighestPeak(frame, frequency, f0Range = (70, 450)):
    sampleRange = (int(frequency / f0Range[1]), int(frequency / f0Range[0]))
    temp = frame[sampleRange[0] : sampleRange[1] + 1]
    return np.argmax(temp) + sampleRange[0]

# Tìm F0 của 1 frame
def getPitch(frame, frequency, threshold = 0.446):
    posOfPeak = getHighestPeak(frame, frequency)
    if frame[posOfPeak] >= threshold:
        return frequency / posOfPeak
    return 0

def getVoicedFrame(framesACFArray, frequency, input, threshold = 0.446):
    VoicedFrame = []
    input = input[:-2]
    j = 0 
    for i in range(len(framesACFArray)):
        posOfPeak = getHighestPeak(framesACFArray[i], frequency)
        if input[j][2] == 'v':
            if framesACFArray[i][posOfPeak] >= threshold:
                VoicedFrame.append(framesACFArray[i])
        else:
            j += 1
    return VoicedFrame

def getUnvoicedFrame(framesACFArray, frequency, input, thresholdV = 0.446, thresholdU = 0.15):
    UnvoicedFrame = []
    input = input[:-2]
    j = 0 
    for i in range(len(framesACFArray)):
        posOfPeak = getHighestPeak(framesACFArray[i], frequency)
        if input[j][2] == 'uv':
            if framesACFArray[i][posOfPeak] >= thresholdU and framesACFArray[i][posOfPeak] < thresholdV:
                UnvoicedFrame.append(framesACFArray[i])
        else:
            j += 1
    return UnvoicedFrame

# Main
# Đọc tín hiệu mẫu
index = 0
for i in range(0, len(FILE_WAV_THKT)):
    frequency, signal = read(join(FILE_PATH_THKT, FILE_WAV_THKT[i]))
    # print("Frequency : ", frequency)
    # print("Signal : ", signal)

    frameLength = int(TIME_FRAME * frequency) # Độ dài của 1 frame (đơn vị mẫu)
    framesArray = getFramesArray(signal, frameLength)
    framesACFArray = [ACF(framesArray[i], frameLength) for i in range(len(framesArray))]
    # Tính ngưỡng riêng của từng tín hiệu
    # meanV, stdV, meanU, stdU = findThreshold(framesACFArray, frequency, readFileInput(FILE_LAB_THHL[i]))
    F0 = np.zeros(len(framesArray))
    timeSample = np.zeros(len(framesArray))
    for i in range(len(framesACFArray)):
        F0[i] = getPitch(framesACFArray[i], frequency)
        # F0[i] = getPitch(framesACFArray[i], frequency, meanV)
        timeSample[i] = TIME_FRAME * i / 2  

    F0mean = np.mean([i for i in F0 if i > 0 and i < 450])
    F0std = np.std([i for i in F0 if i > 0 and i < 450])

    # Show
    plt.figure(index + 1)
    plt.subplot(4, 1, 1)
    plt.title(f"Signal: {FILE_WAV_THKT[index]}")
    plt.plot(signal)
    plt.subplot(4, 1, 2)
    plt.title(f"F0 ACF - F0mean = {F0mean}, F0std = {F0std}")
    plt.ylim([0, 450])
    plt.plot(timeSample, F0, '.')
    VoicedACF = getVoicedFrame(framesACFArray, frequency, readFileInput(FILE_LAB_THKT[index]))
    UnvoicedACF = getUnvoicedFrame(framesACFArray, frequency, readFileInput(FILE_LAB_THKT[index]))
    plt.subplot(4, 1, 3)
    plt.title("Voiced ACF")
    # plt.plot(VoicedACF[index])
    plt.plot(VoicedACF[2])
    plt.plot()
    plt.subplot(4, 1, 4)
    plt.title("Unvoiced ACF")
    # plt.plot(UnvoicedACF[index])
    plt.plot(UnvoicedACF[1])
    plt.plot()
    plt.tight_layout()
    index += 1
plt.show()