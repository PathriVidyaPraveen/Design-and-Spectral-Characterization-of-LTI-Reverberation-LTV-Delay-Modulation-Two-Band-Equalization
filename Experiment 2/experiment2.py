import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

ALPHA = 0.9
D0_MS = 3.0
DELTA_MS = 2.0
FM_HZ = 0.5

FM_SWEEP = [0.1,0.5,2.0,5.0]
DELTA_SWEEP = [0.5,1.0,2.0,3.0]

NFFT = 8192 # FFT length per frame for magnitude spectrum

def load(path):
    x,fs = sf.read(path,always_2d=True)
    x = x.mean(axis=1)
    return 0.5*x/ np.abs(x).max(), fs

def time_varying_delay(x,fs,alpha,d0_ms,delta_ms,fm_hz):
    n = np.arange(len(x))
    D = (d0_ms + delta_ms*np.sin(2*np.pi*fm_hz*n / fs))*1e-3*fs # samples
    t=n-D
    k=np.floor(t).astype(int)
    mu = t-k
    pad = int(np.ceil(D.max())) + 1
    xp = np.concatenate([np.zeros(pad), x, np.zeros(1)])
    x_delayed = (1-mu)*xp[k+pad] + mu*xp[k+pad+1]

    return x + alpha*x_delayed

def fft(frames):
    X = frames[:,None,:].astype(complex)
    while X.shape[2] > 1:
        half = X.shape[2]//2
        E,O = X[:,:,:half],X[:,:,half:]
        M = X.shape[1]
        W = np.exp(-1j*np.pi*np.arange(M)/M)[None,:,None]
        X = np.concatenate([E + W*O, E - W*O], axis=1)
    return X[:,:,0]

def magnitude_spectrum(x,fs,nfft=NFFT):
    hop = nfft//2
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(nfft)/nfft)
    frames = np.stack([x[s:s+nfft]*w for s in range(0,len(x)-nfft+1,hop)])
    mag = np.abs(fft(frames))[:,:nfft//2+1].mean(axis=0)
    f = np.arange(nfft//2+1)*fs/nfft
    return f, 20*np.log10(mag+1e-12)

def plot_waveforms(x,y,fs,label,path):
    t = np.arange(len(x))/fs
    c = np.argmax(np.abs(x))
    z = slice(max(c-int(0.015*fs),0), c+int(0.015*fs))
    fig,ax = plt.subplots(2,1,figsize=(10,6))
    for a,s in zip(ax,[slice(None),z]):
        a.plot(t[s],x[s],lw=0.6,label="original")
        a.plot(t[s],y[s],lw=0.6,alpha=0.7,label=label)
        a.set_xlabel("time (s)")
        a.set_ylabel("amplitude")
        a.legend(loc="upper right")
    ax[0].set_title("full clip")
    ax[1].set_title("zoom: 30 ms around the largest peak")
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)

def plot_spectra(x,outputs,fs,title,path):
    plt.figure(figsize=(10,5))
    f,X = magnitude_spectrum(x,fs)
    plt.semilogx(f[1:],X[1:],lw=1,color="k",label="original")
    for label,y in outputs.items():
        f,Y = magnitude_spectrum(y,fs)
        plt.semilogx(f[1:],Y[1:],lw=0.8,label=label)
    plt.xlabel("frequency (Hz)"); plt.ylabel("magnitude (dB)")
    plt.title(title)
    plt.xlim(20,fs/2)
    plt.grid(which="both",alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=150)
    plt.close()

if __name__ == "__main__":
    x,fs = load("../recording.wav")
    y = time_varying_delay(x,fs,ALPHA,D0_MS,DELTA_MS,FM_HZ)
    sf.write("original.wav",x,fs)
    sf.write("exp2_ltv.wav",y,fs)

    fm_outputs = {}
    for fm in FM_SWEEP:
        y_fm = time_varying_delay(x,fs,ALPHA,D0_MS,DELTA_MS,fm)
        sf.write(f"exp2_fm_{fm}Hz.wav",y_fm,fs)
        fm_outputs[f"fm = {fm} Hz"] = y_fm
 
    delta_outputs = {}
    for delta in DELTA_SWEEP:
        y_delta = time_varying_delay(x,fs,ALPHA,D0_MS,delta,FM_HZ)
        sf.write(f"exp2_delta_{delta}ms.wav",y_delta,fs)
        delta_outputs[f"dD = {delta} ms"] = y_delta
 
    # waveforms and magnitude spectra vs the original
    params = f"D0 = {D0_MS} ms, dD = {DELTA_MS} ms, fm = {FM_HZ} Hz, alpha = {ALPHA}"
    plot_waveforms(x,y,fs,"LTV output","exp2_waveform.png")
    plot_spectra(x,{"LTV output":y},fs,params,"exp2_spectrum.png")
    plot_spectra(x,fm_outputs,fs,f"fm sweep (dD = {DELTA_MS} ms)","exp2_spectrum_fm_sweep.png")
    plot_spectra(x,delta_outputs,fs,f"dD sweep (fm = {FM_HZ} Hz)","exp2_spectrum_delta_sweep.png")