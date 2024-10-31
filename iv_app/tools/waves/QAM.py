import numpy as np
import matplotlib.pyplot as plt

def binary_to_qam(binary_data, symbols_per_second=10, samples_per_symbol=20):
    # Convert binary string to array of integers
    bits = np.array([int(b) for b in binary_data])
    # Pad the binary data to make it divisible by 4 (for 16-QAM)
    padding = (4 - (len(bits) % 4)) % 4
    bits = np.pad(bits, (0, padding))

    # Reshape into groups of 4 bits
    bit_groups = bits.reshape(-1, 4)

    # Create amplitude mapping for 16-QAM
    # Using Gray coding for better error resistance
    amplitude_map = {
    (0,0,0,0): (-3-3j),(0,0,0,1): (-3-1j),(0,0,1,1): (-3+3j),(0,0,1,0): (-3+1j),
    (0,1,0,0): (-1-3j),(0,1,0,1): (-1-1j),(0,1,1,1): (-1+3j),(0,1,1,0): (-1+1j),
    (1,1,0,0): (3-3j),(1,1,0,1): (3-1j),(1,1,1,1): (3+3j),(1,1,1,0): (3+1j),
    (1,0,0,0): (1-3j),(1,0,0,1): (1-1j),(1,0,1,1): (1+3j),(1,0,1,0): (1+1j)
    }

    # Convert bit groups to complex amplitudes
    symbols = np.array([amplitude_map[tuple(group)] for group in bit_groups])

    # Create time array
    t = np.linspace(0, len(symbols), len(symbols) * samples_per_symbol)

    # Create carrier waves (in-phase and quadrature)
    carrier_freq = symbols_per_second
    i_carrier = np.cos(2 * np.pi * carrier_freq * t)
    q_carrier = np.sin(2 * np.pi * carrier_freq * t)

    # Upsample symbols to match carrier wave
    i_symbols = np.repeat(symbols.real, samples_per_symbol)
    q_symbols = np.repeat(symbols.imag, samples_per_symbol)

    # Modulate carriers
    signal = i_symbols * i_carrier - q_symbols * q_carrier

    return t, signal, symbols

# Example usage
# binary_data = ("0000"
#                "0001"
#                "0010"
#                "0011"
#                "0100"
#                "0101"
#                "0110"
#                "0111"
#                "1000"
#                "1001"
#                "1010"
#                "1011"
#                "1100"
#                "1101"
#                "1110"
#                "1111")
# binary_data = ("0000"
#                "0100"
#                "1000"
#                "1100"
#                "0001"
#                "0101"
#                "1001"
#                "1101"
#                "0010"
#                "0110"
#                "1010"
#                "1110"
#                "0011"
#                "0111"
#                "1011"
#                "1111")
binary_data = ("0101"
               "0100"
               "0111"
               "0110"
               "0100"
               "1000"
               "0001"
               "1101"
               "1100"
               "0011"
               "0000"
               "1111"
               )
# binary_data = "1011001110101010100101010010101001001001011100010010100100101001000010010010010111001010100100111000"
t, signal, symbols = binary_to_qam(binary_data)

# Create visualization
plt.figure(figsize=(15, 10))

# Plot time domain signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('16-QAM Modulated Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot constellation diagram
plt.subplot(2, 1, 2)
plt.scatter(symbols.real, symbols.imag)
plt.title('Constellation Diagram')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.axis('equal')

print(binary_data)
plt.tight_layout()
plt.show()