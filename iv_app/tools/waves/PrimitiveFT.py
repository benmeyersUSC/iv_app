import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def makeFrequencyMatrix(ran, duration, sampleRate):
    time_points = np.arange(0, duration, 1 / sampleRate)
    frequencies = []
    for freq in range(1, ran + 1):
        signal = np.sin(2 * np.pi * freq * time_points)
        frequencies.append(signal)
    return pd.DataFrame(frequencies)


def runNewWave(numFreqs, freqs=None, ran=100, sampleRate=3000, duration=.549):
    if freqs is None:
        freqs = []
    lenFreqs = len(freqs)
    if numFreqs < 1:
        numFreqs = 1
    elif lenFreqs >= numFreqs:
        numFreqs = lenFreqs

    if not freqs and numFreqs > 0:
        freqs = [int(np.random.random() * 100) for _ in range(numFreqs)]

    frequency_matrix = makeFrequencyMatrix(ran, duration, sampleRate)
    time_points = np.arange(0, duration, 1 / sampleRate)

    usedFreqs = []
    sigs = []
    signal = 0
    for i in range(numFreqs):
        if i < lenFreqs:
            x = freqs[i]
        else:
            x = int(np.random.random() * ran)
        amplitude = 1 + (np.random.random() * 0.5 - 0.1)
        usedFreqs.append([x, amplitude])
        signal += amplitude * np.sin(2 * np.pi * x * time_points)
        sigs.append(x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [0.65, 0.35]})

    # Plot for the synthetic wave and component sine waves
    ax1.plot(time_points * 1000, signal, label="Synthetic Wave", linewidth=2.7)
    for f in usedFreqs:
        ax1.plot(time_points * 1000, f[1] * np.sin(2 * np.pi * f[0] * time_points), alpha=0.36, label=f"{f[0]} Hz")
    ax1.set_xlim(0, duration * 1000)
    ax1.set_ylim(min(signal), max(signal))
    ax1.set_xlabel('Time (ms)')
    # ax1.legend()
    num_lines = len(usedFreqs) + 1  # +1 for the main "Synthetic Wave" line
    max_rows = 3  # Limit to at most 3 rows

    # Calculate the required number of columns to keep the legend to 3 rows max
    ncol = math.ceil(num_lines / max_rows)

    # Set up the legend with the calculated number of columns
    ax1.legend(ncol=ncol, loc='upper right', bbox_to_anchor=(1, 1))

    # Plot for Fourier transform result
    ft = signal @ frequency_matrix.T
    ax2.plot(ft)
    ax2.set_title(f"Computed Frequencies (Hz): {[i for i in sigs]}")
    ax2.set_xlabel('Frequency (Hz)')

    # Add titles and save the plot
    plt.suptitle("Fourier Transforms")
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(os.getcwd(), 'static', 'images', 'fourier')
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(save_dir, 'fourier.png')
    plt.savefig(file_path)


def main():
    runNewWave(3, [99])


if __name__ == '__main__':
    main()