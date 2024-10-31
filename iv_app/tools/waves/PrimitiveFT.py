import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile

def makeFrequencyMatrix(ran, duration, sampleRate):
    """
    Create a DataFrame where each row is a sine wave at a unique frequency.

    Parameters:
    - ran: int - Number of rows/frequencies up to which to generate sine waves.
    - duration: float - Total time in seconds of the signal.
    - sampleRate: float - Number of samples per second (sampling rate).

    Returns:
    - DataFrame where each row is a sine wave at a given frequency.
    """
    time_points = np.arange(0, duration, 1 / sampleRate)
    frequencies = []

    for freq in range(1, ran + 1):
        # Generate the sine wave for the current frequency
        signal = np.sin(2 * np.pi * freq * time_points)
        frequencies.append(signal)
        # if freq == ran//2:
        #     plt.plot(signal)
        #     plt.show()

    # Convert the list of signals into a DataFrame
    return pd.DataFrame(frequencies)

def runNewWave(numFreqs, ran=100, sampleRate=1000, duration=5):
    if numFreqs < 1:
        numFreqs = 1
    frequency_matrix = makeFrequencyMatrix(ran, duration, sampleRate)
    print(frequency_matrix)
    time_points = np.arange(0, duration, 1 / sampleRate)

    first = int(np.random.random() * ran)
    sigs = [first]
    signal = np.sin(2 * np.pi * first * time_points)
    for i in range(numFreqs-1):
        x = int(np.random.random() * ran)
        signal += np.sin(2 * np.pi * x * time_points)
        sigs.append(x)

    plt.plot(signal)
    plt.show()

    ft = signal @ frequency_matrix.T
    plt.plot(ft)
    plt.title(f"Frequencies (Hz): {[i for i in sigs]}")
    plt.show()

def plotFromWav(file='/Users/benmeyers/Desktop/homnee.wav'):
    sample_rate, data = wavfile.read(file)

    # If the data is stereo, select only one channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Generate a time axis based on the sample rate
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    plt.plot(data)
    plt.show()

    return data




def main():

    runNewWave(3)



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
#
#
# def makeFrequencyMatrix(ran, num_samples, sampleRate):
#     """
#     Create a DataFrame where each row is a sine wave at a unique frequency.
#     Works with both synthetic and real signals.
#
#     Parameters:
#     - ran: int - Number of rows/frequencies up to which to generate sine waves.
#     - num_samples: int - Number of samples in the signal
#     - sampleRate: float - Number of samples per second (sampling rate)
#     """
#     duration = num_samples / sampleRate
#     time_points = np.linspace(0, duration, num_samples)
#     frequencies = []
#
#     for freq in range(1, ran + 1):
#         signal = np.sin(2 * np.pi * freq * time_points)
#         frequencies.append(signal)
#     df = pd.DataFrame(frequencies)
#     print(f"Frequency Matrix: {df}")
#     return df
#
#
# def runNewWave(numFreqs, ran=100, sampleRate=44100, duration=0.1):
#     """
#     Generate and analyze a synthetic signal with random frequency components.
#     Modified to match WAV file parameters by default.
#     """
#     if numFreqs < 1:
#         numFreqs = 1
#
#     num_samples = int(duration * sampleRate)
#     frequency_matrix = makeFrequencyMatrix(ran, num_samples, sampleRate)
#     time_points = np.linspace(0, duration, num_samples)
#
#     # Generate synthetic signal with random frequencies
#     first = int(np.random.random() * ran)
#     sigs = [first]
#     signal = np.sin(2 * np.pi * first * time_points)
#     for i in range(numFreqs - 1):
#         x = int(np.random.random() * ran)
#         signal += np.sin(2 * np.pi * x * time_points)
#         sigs.append(x)
#
#     plt.figure(figsize=(12, 4))
#     plt.plot(signal)
#     plt.title("Synthetic Signal")
#     plt.show()
#
#     # Analyze frequency components
#     ft = signal @ frequency_matrix.T
#     plt.figure(figsize=(12, 4))
#     plt.plot(ft)
#     plt.title(f"Frequencies (Hz): {[i for i in sigs]}")
#     plt.show()
#
#     return signal, ft, frequency_matrix
#
#
# def analyze_wav(file_path, max_freq=100):
#     """
#     Analyze frequencies in a WAV file using our frequency matrix approach.
#     """
#     # Read the WAV file
#     sample_rate, data = wavfile.read(file_path)
#
#     # If stereo, take only one channel
#     if len(data.shape) > 1:
#         data = data[:, 0]
#
#     # Normalize the signal to match sine wave amplitude range (-1 to 1)
#     data = data.astype(float)
#     data = data / np.max(np.abs(data))
#
#     # Create frequency matrix matching the length of our signal
#     num_samples = len(data)
#     frequency_matrix = makeFrequencyMatrix(max_freq, num_samples, sample_rate)
#
#     # Plot original signal
#     plt.figure(figsize=(12, 4))
#     plt.plot(data)
#     plt.title("Original WAV Signal")
#     plt.show()
#
#     # Compute frequency components
#     ft = data @ frequency_matrix.T
#
#     # Plot frequency components
#     plt.figure(figsize=(12, 4))
#     plt.plot(np.arange(max_freq), np.abs(ft))
#     plt.title("Frequency Components")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.show()
#
#     # Find dominant frequencies
#     dominant_freqs = np.argsort(np.abs(ft))[-5:][::-1]  # Top 5 frequencies
#     print(f"Dominant frequencies (Hz): {dominant_freqs}")
#
#     return data, ft, frequency_matrix
#
#
# def reconstruct_signal(ft, frequency_matrix):
#     """
#     Reconstruct the signal using all frequency components.
#     """
#     return ft @ frequency_matrix
#
#
# def main():
#     # Example 1: Generate and analyze synthetic signal
#     print("Generating synthetic signal...")
#     synthetic_signal, synthetic_ft, synthetic_matrix = runNewWave(5)
#
#     # # Example 2: Analyze WAV file
#     # print("\nAnalyzing WAV file...")
#     # file_path = '/Users/benmeyers/Desktop/homnee.wav'
#     # wav_signal, wav_ft, wav_matrix = analyze_wav(file_path, max_freq=100)
#     #
#     # # Reconstruct both signals
#     # synthetic_reconstructed = reconstruct_signal(synthetic_ft, synthetic_matrix)
#     # wav_reconstructed = reconstruct_signal(wav_ft, wav_matrix)
#     #
#     # # Plot reconstructions
#     # plt.figure(figsize=(12, 8))
#     #
#     # plt.subplot(221)
#     # plt.plot(synthetic_signal)
#     # plt.title("Original Synthetic Signal")
#     #
#     # plt.subplot(222)
#     # plt.plot(synthetic_reconstructed)
#     # plt.title("Reconstructed Synthetic Signal")
#     #
#     # plt.subplot(223)
#     # plt.plot(wav_signal)
#     # plt.title("Original WAV Signal")
#     #
#     # plt.subplot(224)
#     # plt.plot(wav_reconstructed)
#     # plt.title("Reconstructed WAV Signal")
#     #
#     # plt.tight_layout()
#     # plt.show()


if __name__ == '__main__':
    main()
