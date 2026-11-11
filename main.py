import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

fs = 1000
T_total = 2.0
N = int(fs * T_total)
t = np.arange(N) / fs
A = 1.0
phi = 0.0
f_mod = 1.0
f_carrier = 8.0

m = (np.sin(2 * np.pi * f_mod * t + phi) >= 0).astype(float)
carrier = np.sin(2 * np.pi * f_carrier * t + phi)

def compute_spectrum(signal, fs):
    N = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs_spectrum = np.fft.rfftfreq(N, 1/fs)
    amplitude = np.abs(spectrum) / N
    amplitude[1:-1] *= 2
    return freqs_spectrum, amplitude, spectrum

s_am = (0.5 + m) * carrier

f_inst = f_carrier + 10 * m
s_fm = np.cos(2 * np.pi * np.cumsum(f_inst) / fs)

s_pm = np.cos(2 * np.pi * f_carrier * t + np.pi * m)

for name, signal in [("AM", s_am), ("FM", s_fm), ("PM", s_pm)]:
    plt.figure(figsize=(8, 4))
    plt.plot(t, signal, color="blue")
    plt.title(f"Сигнал с {name}-модуляцией")
    plt.xlabel("Время, с")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.0)
    plt.ylim(-2, 2)
    plt.tight_layout()
    plt.savefig(f"time_{name}.png")
    plt.close()

for name, signal in [("AM", s_am), ("FM", s_fm), ("PM", s_pm)]:
    f_axis, amp, _ = compute_spectrum(signal, fs)
    plt.figure(figsize=(8, 4))
    plt.stem(f_axis, amp, linefmt="b-", markerfmt="bo", basefmt=" ")
    plt.title(f"Спектр {name}-сигнала")
    plt.xlabel("Частота, Гц")
    plt.ylabel("Амплитуда")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(f"spectrum_{name}.png")
    plt.close()

f_axis, amp_am, spec_am = compute_spectrum(s_am, fs)
spec_cut = spec_am.copy()
mask = (f_axis >= f_carrier - 2) & (f_axis <= f_carrier + 2)
spec_cut[~mask] = 0
s_synth = np.fft.irfft(spec_cut, n=N)

plt.figure(figsize=(8, 4))
plt.plot(t, s_synth, color="green")
plt.title("Синтезированный AM-сигнал (обрезанный спектр)")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True, alpha=0.3)
plt.xlim(0, 2.0)
plt.ylim(-2, 2)
plt.tight_layout()
plt.savefig("synth_AM.png")
plt.close()

analytic = hilbert(s_synth)
envelope = np.abs(analytic)

plt.figure(figsize=(8, 4))
plt.plot(t, envelope, color="orange", label="Огибающая")
plt.plot(t, m, color="red", linestyle="--", label="Исходный меандр")
plt.title("Огибающая синтезированного сигнала")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True, alpha=0.3)
plt.xlim(0, 2.0)
plt.ylim(-0.1, 2)
plt.legend()
plt.tight_layout()
plt.savefig("envelope_AM.png")
plt.close()

binary = (envelope > 0.5).astype(float)

plt.figure(figsize=(8, 4))
plt.plot(t, binary, color="purple", label="После компаратора")
plt.plot(t, m, color="red", linestyle="--", label="Исходный меандр")
plt.title("Восстановленный модулирующий сигнал")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")
plt.grid(True, alpha=0.3)
plt.xlim(0, 2.0)
plt.ylim(-0.1, 2)
plt.legend()
plt.tight_layout()
plt.savefig("filtered_AM.png")
plt.close()