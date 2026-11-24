import numpy as np
import math
import soundfile as sf
import matplotlib.pyplot as plt


def low_pass_filter(x, fc, fs):
    # 2nd order Butterworth LPF (biquad) 
    # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2] is the diff equation
    w0 = 2*math.pi*fc/fs
    Q = 1/math.sqrt(2)
    alpha = math.sin(w0)/(2*Q)
    cosw = math.cos(w0)
    # coefficients
    a0 = 1 + alpha
    b0 = (1 - cosw)/(2*a0)
    b1 = (1 - cosw)/a0
    b2 = (1 - cosw)/(2*a0)
    a1 = -2*cosw/a0
    a2 = (1 - alpha)/a0
    y = np.zeros(len(x))
    x1 = x2 = 0.0 # initialized to 0s for prev inputs
    y1 = y2 = 0.0 # initialized to 0s for prev outputs

    for n in range(len(x)):
        y[n] = (b0*x[n])+(b1*x1)+(b2*x2)-(a1*y1)-(a2*y2)
        x2, x1 = x1, x[n] # shifting input history
        y2, y1 = y1, y[n] #shifting output history
    return y


def save_safe(name, sig, fs):
    # scale down only if the signal would clip (|sample| > 1)
    peak = np.abs(sig).max()
    if peak > 1.0:
        sig = sig/peak*0.99
    sf.write(name, sig, fs)


# task 1 - low band from filter and highband we got by subtracting low band signal from original signal
x,fs = sf.read("audio/input.wav")
fc = 500 # cutoff fre in hz

if x.ndim == 1: # for mono signal
    y1 = low_pass_filter(x, fc, fs)
else: # we filter each channel separately in case of stereo signal
    yl = low_pass_filter(x[:,0], fc, fs)# left channel
    yr = low_pass_filter(x[:,1], fc, fs) # right channel
    y1 = np.column_stack([yl,yr]) # join back into (N, 2)

y2 = x-y1  # high = original - low

print("max |x - (y1 + y2)| =", np.abs(x - (y1 + y2)).max())

save_safe("y11_low_part.wav", y1, fs)
save_safe("y22_high_part.wav", y2, fs)


# implemnted 4 modes and multiplexer
def eq(gL, gH):
    return gL*y1 + gH*y2 # same eqn in doc ie y = gL*xL + gH*xH

# we tuned this experimentally by listening to audio
y_drum = eq(2.0, 1.0) 
y_guitar = eq(1.0, 1.5) 
y_both = eq(2.0, 1.5)

outs = [x, y_drum, y_guitar, y_both] 
names = ["original", "drum", "guitar", "drum_guitar"]


def mux(mode):
    return outs[mode] 


for mode in range(4):
    print("mode", mode, names[mode], ": peak =", round(float(np.abs(mux(mode)).max()), 2))


# task 3
peak = max(np.abs(o).max() for o in outs)
if peak > 1.0 :
    scale = 0.99/peak
else :
    scale = 1.0

mode = 3
chosen = mux(mode)
print("selected mode:", names[mode])

for m in range(4) :
    sf.write("exp3_" + names[m] + ".wav", mux(m) * scale, fs)
    print("saved exp3_" + names[m] + ".wav")

# task 4 - magnitude spectra of original and processed signals

def dft(x):
    x = np.asarray(x)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    X = W @ x
    return X

def magnitude_spectrum(X):
    return np.abs(X)

N = 2048 # dft length
seg_start = len(x) // 2
win = np.hanning(N) 
f = np.arange(N//2+1)*fs/N 

plt.figure(figsize=(9, 5))
for m in range(4):
    sig = mux(m)
    if sig.ndim > 1: # stereo to mono 
        sig = sig.mean(axis=1)
    seg = sig[seg_start : seg_start + N] * win
    mag = magnitude_spectrum(dft(seg))[: N // 2 + 1]
    plt.semilogx(f[1:], 20 * np.log10(mag[1:] + 1e-12), label=names[m], alpha=0.8)

plt.axvline(fc, color="k", ls="--", lw=0.8, label="cutoff fc")
plt.title("Magnitude spectra of original and processed signals")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_spectra.png", dpi=150)
plt.close()  


# experimented with cutoffs and gains

def run_eq(fc_try, gL, gH, tag) :
    if x.ndim == 1 :
        low = low_pass_filter(x, fc_try, fs)
    else : 
        low = np.column_stack([low_pass_filter(x[:, c], fc_try, fs) for c in range(x.shape[1])])
    high = x - low 
    y = gL*low + gH*high
    save_safe("exp5_" + tag + ".wav", y, fs)
    print(tag, ": fc =", fc_try, " gL =", gL, " gH =", gH, " peak =", round(float(np.abs(y).max()), 2))

# drum
drum_cutoffs = [400]
drum_gains = [(1.0, 1.0),(2.0, 1.0),(2.5,1.0)] # (gL,gH)

for fc_try in drum_cutoffs :
    for gL, gH in drum_gains :
        run_eq(fc_try, gL, gH, "fc" + str(fc_try) + "_drum_gL" + str(gL) + "_gH" + str(gH))

# guitar 
guitar_cutoffs = [800]
guitar_gains = [ (1.0, 1.0),(1.0, 2.0),(1.0,2.5)]

for fc_try in guitar_cutoffs :
    for gL, gH in guitar_gains :
        run_eq(fc_try, gL, gH, "fc" + str(fc_try) + "_guitar_gL" + str(gL) + "_gH" + str(gH))

        