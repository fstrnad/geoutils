from importlib import reload
import climnet.utils.general_utils as gut
import climnet.utils.statistic_utils as sut
from statsmodels.tsa.ar_model import AutoReg
import xarray as xr
import scipy.signal as sig
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd
import climnet.tsa.filters as flt
import climnet.utils.time_utils as tu
from tqdm import tqdm


def phi_hilbert_transform(data):
    ht = sig.hilbert(data)
    imag = np.imag(ht)
    real = np.real(ht)
    phi = np.unwrap(np.angle(ht))
    return {'phi': phi, 'im': imag, 're': real}


def phase_hilbert(data, cutoff=1, anomaly='dayofyear'):
    # estimate climate anomalies
    print('Climate Anomalies...')
    data = tu.compute_anomalies(data, group=anomaly)

    # filter the time series
    print('Filter time series...')
    data_flt = flt.apply_butter_filter(ts=data, cutoff=cutoff)

    print("time series derivatives ...")
    data_der = np.gradient(data_flt, 3)

    print('Hilbert Transform...')
    phi_dict = phi_hilbert_transform(data=data_der)

    phi_dict['der'] = data_der

    return phi_dict


def phase_coherence(data1, data2, cutoff=1, anomaly='dayofyear', perc=10):
    if len(data1) != len(data2):
        print(f'{len(data1)} != {len(data2)}')
        raise ValueError(f'Time Series not of same length!')
    times = data1.time
    phi1_dict = phase_hilbert(data=data1, cutoff=cutoff, anomaly=anomaly)
    phi2_dict = phase_hilbert(data=data2, cutoff=cutoff, anomaly=anomaly)
    phi1 = phi1_dict['phi']
    phi2 = phi2_dict['phi']

    pclo = 100. - perc
    pchi = perc

    print("instantaneous phase difference ...")
    del_phi = phi2 - phi1
    del_phi_dot = np.gradient(del_phi, 3)
    only_neg = del_phi_dot[del_phi_dot < 0.]
    lothres = np.percentile(only_neg, pclo)
    only_pos = del_phi_dot[del_phi_dot >= 0.]
    hithres = np.percentile(only_pos, pchi)
    # Find phase coherent indices
    print("find coherent phases ...")
    pc_idx = (del_phi_dot >= lothres) * (del_phi_dot <= hithres)
    te = times[pc_idx]

    del_phi = tu.create_xr_ts(data=del_phi, times=times)

    return {'te': te,
            'pc_idx': pc_idx,
            'del_phi': del_phi,
            'phi1': phi1,
            'phi2': phi2,
            'im1': phi1_dict['im'],
            'im2': phi2_dict['im']
            }
