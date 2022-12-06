#! /usr/local/opt/python/libexec/bin/python
#%%
"""
Use the functions phaselock.py to estimate phase locking between ISMR & NINO3
=============================================================================


"""
# Created: Thu Oct 12, 2017  02:37PM
# Last modified: Fri Nov 17, 2017  09:12AM
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import sys
import numpy as np
from scipy import signal
import utils
import figs


if __name__ == "__main__":
    verbose = True

    # load data and create common time axis array
    utils.printmsg("load data ...", verbose)
    dnino, dismr = utils.load_data()
    time = utils.common_time_axis(dismr)

    # estimate climate anomalies
    utils.printmsg("climate anomalies ...", verbose)
    nino_anom = utils.climate_anomaly(dnino, time)
    ismr_anom = utils.climate_anomaly(dismr, time)

    # filter the time series
    utils.printmsg("filtered time series ...", verbose)
    nino_filt = utils.iirfilter(nino_anom, N=5, pass_freq=1. / 12)
    ismr_filt = utils.iirfilter(ismr_anom, N=5, pass_freq=1. / 12)

    # estimate the derivative of the filtered time series
    utils.printmsg("time series derivatives ...", verbose)
    nino_grad = np.gradient(nino_filt, 3)
    ismr_grad = np.gradient(ismr_filt, 3)

    # use the Hilbert transform to estimate the phase of the signal
    utils.printmsg("Hilbert transform ...", verbose)
    nino_phi = np.unwrap(np.angle(signal.hilbert(nino_grad)))
    ismr_phi = np.unwrap(np.angle(signal.hilbert(ismr_grad)))

    # calculate the instantaneous phase difference
    utils.printmsg("instantaneous phase difference ...", verbose)
    del_phi = ismr_phi - nino_phi

    # compute gradient and get event series for times when gradient << 1
    utils.printmsg("periods of phase locking ...", verbose)
    perc = 10
    pclo = 100. - perc
    pchi = perc
    del_phi_dot = np.gradient(del_phi, 3)
    only_neg = del_phi_dot[del_phi_dot < 0.]
    lothres = np.percentile(only_neg, pclo)
    only_pos = del_phi_dot[del_phi_dot >= 0.]
    hithres = np.percentile(only_pos, pchi)
    ct = time.copy()
    idx = (del_phi_dot >= lothres) * (del_phi_dot <= hithres)
    te = ct[idx]

    # plot figures
    utils.printmsg("output figures ...", verbose)
    figs.input_timeseries(time,
                          {# nino_dat
                             "anom": nino_anom,
                             "filt": nino_filt,
                             "grad": nino_grad,
                          },
                          {# ismr_dat
                             "anom": ismr_anom,
                             "filt": ismr_filt,
                             "grad": ismr_grad,
                          },
                          )
    figs.delphi_timeseries(ct, del_phi, te)
    figs.delphi_histogram(del_phi_dot, lothres, hithres)
    # %%
