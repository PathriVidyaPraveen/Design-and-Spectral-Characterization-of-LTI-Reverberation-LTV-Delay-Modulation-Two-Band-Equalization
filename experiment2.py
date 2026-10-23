import numpy as np
import soundfile as sf

ALPHA = 0.9
D0_MS = 3.0
DELTA_MS = 2.0
FM_HZ = 0.5

def load(path):
    x,fs = sf.read(path,always_2d=True)
    x = x.mean(axis=1)
    return 0.5*x/ np.abs(x).max(), fs

def time_varying_delay(x,fs,alpha,d0_ms,delta_ms,fm_hz):
    n = np.arange(len(x))
    D = (d0_ms + delta_ms*np.sin(2*np.pi*fm_hz*n / fs))*1e-3*fs # samples
    x_delayed = np.interp(n-D,n,x,left=0)

    return x + alpha*x_delayed

if __name__ == "__main__":
    x,fs = load("recording.wav")
    y = time_varying_delay(x,fs,ALPHA,D0_MS,DELTA_MS,FM_HZ)
    sf.write("original.wav",x,fs)
    sf.write("exp2_ltv.wav",y,fs)