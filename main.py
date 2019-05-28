from rtlsdr import RtlSdr
from contextlib import closing
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import spectrogram, windows
from scipy import signal
from skimage.io import imsave, imread
from datetime import datetime
import json
import os
from tqdm import tqdm
import time
from queue import Queue
import asyncio
from pathlib import Path
import warnings

for cat in [RuntimeWarning, UserWarning, FutureWarning]:
    warnings.filterwarnings("ignore", category=cat) 


def split_images(dir="sdr_captures/specs_raw"):
    jpgs = list(Path(dir).rglob('*.jpg'))
    pngs = list(Path(dir).rglob('*.png'))

    img_files = pngs + jpgs
    img_files = list(filter(lambda x: 'chk' not in str(x), img_files))
    for img_file in tqdm(img_files, desc="splitting images"):
        im = imread(img_file)
        shp = list(im.shape)
        shp = list(filter(lambda x: x != 1, shp))
        shp = np.array(shp)
        dim_to_slide_over = shp.argmax()
        win_size = shp[shp.argmin()]
        im_size = shp[dim_to_slide_over]
        for start in range(0, im_size, win_size):
            stop = start + win_size
            if stop >= im_size:
                break
            if dim_to_slide_over == 0:
                chunk = im[start:stop, :]
            elif dim_to_slide_over == 1:
                chunk = im[:, start:stop]
            file_out = str(
                Path(img_file).with_suffix(f".chk_{start}_{stop}.png"))
            imsave(file_out, chunk)


# y -- spectrogram, nf by nt array
# dbf -- Dynamic range of the spectrum

def adjust_dyn_range(x, mx=3, mn=10, rel_to=np.median):
    r = rel_to(x)
    zmax = r+mx
    zmin = r-mn
    x[x<zmin] = zmin
    x[x>zmax] = zmax
    return x


def to_spec(y, fs, fc, NFFT=1024, dbf=60, nperseg=128, normalize=True):

    #w = windows.hamming(nperseg)
    #window = signal.kaiser(nperseg, beta=14)
    
    f, t, y = spectrogram(y, detrend=None, noverlap=int(nperseg/2), nfft=NFFT, fs=fs)

    y = np.fft.fftshift(y, axes=0)
    if normalize:
        #y = norm_spectrum(y)
        y = np.sqrt(np.power(y.real, 2) + np.power(y.imag, 2))
        y = 20 * np.log10(np.abs(y)/ np.abs(y).max())
        y = np.abs(y)
        y = y / y.max()

    
    return y

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def spectrogram(x, fs, fc, m=None, dbf=60):

    if not m:
        m = 1024
    
    isreal_bool = np.isreal(x).all()
    
    lx = len(x);
    nt = (lx + m - 1) // m
    x = np.append(x,np.zeros(-lx+nt*m))
    x = x.reshape((int(m/2),nt*2), order='F')
    x = np.concatenate((x,x),axis=0)
    x = x.reshape((m*nt*2,1),order='F')
    x = x[np.r_[m//2:len(x),np.ones(m//2)*(len(x)-1)].astype(int)].reshape((m,nt*2),order='F')
    xmw = x * windows.hamming(m)[:,None]
    t_range = [0.0, lx / fs]
    if isreal_bool:
        f_range = [ fc, fs / 2.0 + fc]
        xmf = np.fft.fft(xmw,len(xmw),axis=0)
        xmf = xmf[0:m/2,:]
    else:
        f_range = [-fs / 2.0 + fc, fs / 2.0 + fc]
        xmf = np.fft.fftshift( np.fft.fft( xmw ,len(xmw),axis=0), axes=0 )
    f_range = np.linspace(*f_range, xmf.shape[0])
    t_range = np.linspace(*t_range, xmf.shape[1])

    h = xmf.shape[0]
    each = int(h*.10)
    xmf = xmf[each:-each, :]

    xmf = np.sqrt(np.power(xmf.real, 2) + np.power(xmf.imag, 2))
    xmf = np.abs(xmf)
    
    xmf /= xmf.max()

    #throw away sides

    xmf = 20 * np.log10(xmf)
    xmf = np.clip(xmf, -dbf, 0)
    xmf = MinMaxScaler().fit_transform(StandardScaler(with_mean=True, with_std=True).fit_transform(xmf))
    xmf = np.abs(xmf)
    #xmf-=np.median(xmf)
    xmf/=xmf.max()

    
    print(xmf.min(), xmf.max())
    return f_range, t_range, xmf


def append_json(data, path):
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')


async def stream(sdr, N):
    samples_buffer = Queue()
    total = 0
    with tqdm(total=N, desc='sampling') as pbar:
        #for i in range(10):
        #    time.sleep(0.1)

        async for samples in sdr.stream():
            # do something with samples
            # ...

            samples_buffer.put(samples)
            #print(f'put {len(samples)} into buffer')
            total += len(samples)
            pbar.update(len(samples))
            if total >= N:
                break
        # to stop streaming:
        await sdr.stop()

    # done
    sdr.close()
    return samples_buffer


def capture(fc=94.3e6,
            fs=int(1e6),
            gain='auto',
            seconds_dwell=.4
            #offset_dc=5e4
            ):

    
    N = int(seconds_dwell * fs)
    with closing(RtlSdr()) as sdr:
        sdr.sample_rate = fs
        sdr.center_freq = fc# + int(offset_dc)
        sdr.gain = gain
        t = datetime.now()
        stamp = datetime.timestamp(t)

        loop = asyncio.get_event_loop()
        samples_buffer = loop.run_until_complete(stream(sdr, N))

    iq_samples = np.hstack(np.array(list(samples_buffer.queue)))[:N].astype("complex64")
    #iq_samples = shift_mix(iq_samples, -offset_dc, fs)
    #path = os.path.join(out_dir, f'{stamp}.png')
    meta = dict(
        fs=fs,
        fc=fc,
        gain=gain,
        seconds_dwell=seconds_dwell,
        dt_start=stamp
    )



    return iq_samples, meta

def shift_mix(x, hz, fs):
    return x*np.exp(1j*2*np.pi*hz/fs*np.arange(len(x)))


def save_capture(path, spec_img, meta, meta_path):
    imsave(path, spec_img.T)
    append_json(meta, meta_path)


def scan(
        low=80e6,
        high=1000e6,
        repeats=10,
        target_hpb=300,
    ):

    out_dir="sdr_captures/specs_raw"
    meta_path="sdr_captures/dataset.json"

    os.makedirs(out_dir, exist_ok=True)
    for repeat in tqdm(range(repeats), desc='repeats'):
        for fs in [int(3.2e6)]:#list(map(int, (3.2e6, 2e6, 1e6))):
            #for NFFT in [1024, 2048, 2048 * 2]:

            fcs = []
            fc = low
            while fc < high:
                fc += int((fs * (1/3.)))
                fcs.append(fc)
            fcs = np.array(fcs)
            print(f'scanning {len(fcs)} total frequencies...')

            for fc in tqdm(fcs, desc='fcs'):
                try:
                    iq, meta = capture(fc=fc, fs=fs)
                    meta['NFFT'] = closest_power_of_two(fs / target_hpb)
                    meta['hpb'] = fs/meta['NFFT']
                    ff, tt, spec_img = spectrogram(iq, fs, fc, m=meta['NFFT'])
                    
                    img_path = os.path.join(out_dir, f"{meta['dt_start']}.png")
                    save_capture(img_path, spec_img, meta, meta_path)
            


                except Exception as e:
                    print(e)
                    time.sleep(1)
                    pass


def closest_power_of_two(number):
    # Returns next power of two following 'number'
    n = np.ceil(np.log2(number))
    a = np.array([np.power(2, n - 1), np.power(2, n), np.power(2, n + 1)])
    return int(a[np.argmin(np.abs(a - number))])


def norm_spectrum(spec_img):
    spec_img = 20 * np.log10(np.abs(spec_img) / np.max(np.abs(spec_img)))

    mid = np.median(spec_img)
    # high = mid + 30
    # low = mid - 30
    # spec_img[spec_img < low] = low
    # spec_img[spec_img > high] = high

    spec_img = np.abs(spec_img)
    spec_img /= spec_img.max()
    print('spec max:', spec_img.max(), 'spec min:', spec_img.min())
    return spec_img

def parse_measure(s):
    s = s.lower()
    if s[-1].isalpha():
        h, mod = float(s[:-1]), s[-1]
        if mod == 'm':
            h*=1e6
        elif mod == 'k':
            h*=1e3
    else:
        h = int(s)
    return h

def string_to_linspace(s, delim=':'):
    return np.arange(*list(map(parse_measure, s.split(delim))))

#string_to_linspace('24M:28M:3M')


def plot_one(fc=94.3 * 1e6, fs=3e6, target_hpb=300, seconds_dwell=.2):

    NFFT = closest_power_of_two(fs / target_hpb)
    iq_samples, meta = capture(fc=fc, fs=fs, seconds_dwell=seconds_dwell)
    spec_img = to_spec(iq_samples, fs, fc, NFFT=NFFT)
    #spec_img = norm_spectrum(spec_img)
    #spec_img = np.abs(spec_img)
    #spec_img /= spec_img.max()

    #spec_img = 1 - spec_img
    print('img shape:', spec_img.shape)
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.matshow(spec_img.T[:NFFT], cmap=plt.get_cmap('viridis'))
    print(spec_img.T.shape)
    #Wplt.plot(spec_img.T[0, :])
    plt.show()


if __name__ == "__main__":
    #split_images()
    #plot_one()
    scan(repeats=3, target_hpb=1500)
    split_images()
    #plot_one()