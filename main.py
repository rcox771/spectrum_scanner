from rtlsdr import RtlSdr
from contextlib import closing
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from skimage.io import imsave
from datetime import datetime
import json
import os
from tqdm import tqdm
import time
from queue import Queue
import asyncio

# t_range -- time axis label, nt samples
# f_range -- frequency axis label, nf samples
# y -- spectrogram, nf by nt array
# dbf -- Dynamic range of the spectrum


def to_spec(y, fs, fc, NFFT=1024, dbf=60):
    f, t, y = spectrogram(
        y,
        nfft=NFFT,
        fs=fs,
        mode='complex',
    )
    eps = 10.0**(-dbf / 20.0)  # minimum signal

    y = np.sqrt(np.power(y.real, 2) + np.power(y.imag, 2))
    # find maximum
    y = np.abs(y)
    y_max = np.median(y)  #y.median()

    # compute 20*log magnitude, scaled to the max
    y_log = 20.0 * np.log10(y / y_max)

    # rescale image intensity to 256
    #plt.figure()
    #plt.matshow(img)
    #plt.show()
    return y_log


def append_json(data, path):
    with open(path, 'a') as f:
        f.write(json.dumps(data) + '\n')


async def stream(sdr, N):
    samples_buffer = Queue()
    total = 0
    async for samples in sdr.stream():
        # do something with samples
        # ...
        samples_buffer.put(samples)
        print(f'put {len(samples)} into buffer')
        total += len(samples)
        if total >= N:
            break
    # to stop streaming:
    await sdr.stop()

    # done
    sdr.close()
    return samples_buffer


def capture(fc=94.5e6,
            fs=int(1e6),
            gain='auto',
            seconds_dwell=2,
            NFFT=4096,
            out_dir="sdr_captures/specs_raw",
            meta_path="sdr_captures/dataset.json"):

    os.makedirs(out_dir, exist_ok=True)

    with closing(RtlSdr()) as sdr:
        sdr.sample_rate = fs
        sdr.center_freq = fc
        sdr.gain = gain
        t = datetime.now()
        stamp = datetime.timestamp(t)

        loop = asyncio.get_event_loop()
        samples_buffer = loop.run_until_complete(
            stream(sdr, int(seconds_dwell * fs)))

    iq_samples = np.hstack(np.array(list(samples_buffer.queue)))
    print('iq samps: ', iq_samples.shape, iq_samples.dtype)

    #iq_samples = sdr.read_samples_async()
    #time.sleep(1)

    path = os.path.join(out_dir, f'{stamp}.png')
    meta = dict(
        fs=fs,
        fc=fc,
        gain=gain,
        seconds_dwell=seconds_dwell,
        dt_start=stamp,
        NFFT=NFFT,
        path=path)

    spec_img = to_spec(iq_samples, fs, fc, NFFT=NFFT)
    spec_img = np.abs(spec_img)
    spec_img /= spec_img.max()
    #spec_img = 1 - spec_img
    print('img shape:', spec_img.shape)
    imsave(path, spec_img.T)
    append_json(meta, meta_path)


low = 80e6
high = 1000e6
repeats = 10
for repeat in tqdm(range(repeats), desc='repeats'):
    for fs in list(map(int, (1e6, 2e6, 3e6))):
        for NFFT in [1024, 2048, 2048 * 2]:
            fcs = []
            fc = low
            while fc < high:
                fc += int((fs / 5))
                fcs.append(fc)
            fcs = np.array(fcs)
            print(f'scanning {len(fcs)} total frequencies...')

            for fc in tqdm(fcs, desc='fcs'):
                try:
                    capture(fc=fc, fs=fs, NFFT=NFFT)
                except Exception as e:
                    print(e)
                    time.sleep(1)
                    pass
