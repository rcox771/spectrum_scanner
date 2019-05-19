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
            gain=15,
            seconds_dwell=2
            ):

    
    N = int(seconds_dwell * fs)
    with closing(RtlSdr()) as sdr:
        sdr.sample_rate = fs
        sdr.center_freq = fc
        sdr.gain = gain
        t = datetime.now()
        stamp = datetime.timestamp(t)

        loop = asyncio.get_event_loop()
        samples_buffer = loop.run_until_complete(stream(sdr, N))

    iq_samples = np.hstack(np.array(list(samples_buffer.queue)))[:N].astype("complex64")
    
    #path = os.path.join(out_dir, f'{stamp}.png')
    meta = dict(
        fs=fs,
        fc=fc,
        gain=gain,
        seconds_dwell=seconds_dwell,
        dt_start=stamp
    )



    return iq_samples, meta

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
        for fs in list(map(int, (3e6, 2e6, 1e6))):
            #for NFFT in [1024, 2048, 2048 * 2]:

            fcs = []
            fc = low
            while fc < high:
                fc += int((fs / 5))
                fcs.append(fc)
            fcs = np.array(fcs)
            print(f'scanning {len(fcs)} total frequencies...')

            for fc in tqdm(fcs, desc='fcs'):
                try:
                    iq, meta = capture(fc=fc, fs=fs)
                    meta['NFFT'] = closest_power_of_two(fs / target_hpb)
                    meta['hpb'] = fs/meta['NFFT']
                    spec_img = to_spec(iq, fs, fc, NFFT=meta['NFFT'], normalize=True)
                    
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
    scan(repeats=3)
    split_images()
    #plot_one()