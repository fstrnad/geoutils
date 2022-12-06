#! /usr/local/opt/python/libexec/bin/python
"""
Functions used to plot the output figures
=========================================


"""
# Created: Thu Nov 16, 2017  02:05PM
# Last modified: Fri Nov 17, 2017  09:02AM
# Copyright: Bedartha Goswami <goswami@pik-potsdam.de>


import numpy as np
import datetime as dt
from scipy import signal
import matplotlib.pyplot as pl
import matplotlib.dates as mdates


def input_timeseries(time, nino_dat, ismr_dat):
    """
    Plots input time series, filtered time series, and phase space plots.
    """
    # parse the input data
    nino_anom = nino_dat["anom"]
    nino_filt = nino_dat["filt"]
    nino_grad = nino_dat["grad"]
    ismr_anom = ismr_dat["anom"]
    ismr_filt = ismr_dat["filt"]
    ismr_grad = ismr_dat["grad"]

    # set up figure
    fig = pl.figure(figsize=[8.5, 8.5])
    axlabfs, tiklabfs, splabfs = 11, 9, 13

    # set up first axis and plot the NINO index
    ax1 = fig.add_axes([0.10, 0.725, 0.85, 0.210])
    ax1.plot(time, nino_anom,
             c="SteelBlue", label="Original")
    ax1.plot(time, nino_filt,
             c="Tomato",
             label="Filtered")

    # set up second axis and plot the ISMR index
    ax2 = fig.add_axes([0.10, 0.550, 0.85, 0.175])
    ax2.plot(time, ismr_anom,
             c="SteelBlue", label="Original")
    ax2.plot(time, ismr_filt,
             c="Tomato",
             label="Filtered")

    # set up third axis and plot the estimated phases from filtered NINO
    ax3 = fig.add_axes([0.125, 0.10, 0.35, 0.35])
    ax3.plot(nino_grad, np.imag(signal.hilbert(nino_grad)),
             c="Tomato", )

    # set up fourth axis and plot the estimated phases from filtered ISMR
    ax4 = fig.add_axes([0.625, 0.10, 0.35, 0.35])
    ax4.plot(ismr_grad, np.imag(signal.hilbert(ismr_grad)),
             c="Tomato", )

    # prettify ax1 and ax2
    xlo, xhi = dt.datetime(1870, 1, 1), dt.datetime(2015, 12, 31)
    for ax in [ax1, ax2]:
        ax.set_xlim(xlo, xhi)
        XMajorLocator = mdates.YearLocator(base=10, month=6, day=15)
        XMinorLocator = mdates.YearLocator(base=2, month=6, day=15)
        # XMinorLocator = mdates.MonthLocator(bymonthday=15, interval=3)
        XMajorFormatter = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_locator(XMajorLocator)
        ax.xaxis.set_minor_locator(XMinorLocator)
        ax.xaxis.set_major_formatter(XMajorFormatter)
        ax.set_ylim(-3.0, 3.0)
        ax.set_yticks(np.arange(-2.0, 2.01, 1.0))
        ax.grid(which="both")
        ax.tick_params(which="major", axis="both", size=8, direction="out")
        ax.tick_params(which="minor", axis="both", size=5, direction="out")
        ax.tick_params(axis="both", labelsize=tiklabfs)
    leg = ax1.legend(loc="upper right")
    for txt in leg.get_texts():
        txt.set_size(tiklabfs)
    ax1.set_ylim(-4., 4.)
    ax1.tick_params(bottom="off", top="on", which="both",
                    labelbottom="off", labeltop="on")
    ax2.set_xlabel("Time", fontsize=axlabfs)
    ax1.set_xlabel("Time", fontsize=axlabfs)
    ax1.xaxis.set_label_position("top")
    ax1.set_ylabel("Nino 3", fontsize=axlabfs)
    ax2.set_ylabel("ISMR", fontsize=axlabfs)

    # prettify ax3 and ax4
    ax3.set_xlabel("Derivative of filtered Nino 3 signal",
                   fontsize=axlabfs)
    ax3.set_ylabel("Hilbert transform",
                   fontsize=axlabfs)
    ax4.set_xlabel("Derivative of filtered ISMR signal",
                   fontsize=axlabfs)
    ax4.set_ylabel("Hilbert transform",
                   fontsize=axlabfs)
    for ax in [ax3, ax4]:
        ax.grid()
        ax.tick_params(axis="both", labelsize=tiklabfs)
        ax.tick_params(which="major", size=8)

    # save figure
    figname = "../plots/01_input_timeseries.png"
    pl.savefig(figname)
    print("figure saved to: %s" % figname)

    return None


def delphi_timeseries(ct, del_phi, te):
    """
    Plots the instantaneous phase diff with periods of phase sync highlighted.
    """
    # set up figure
    fig = pl.figure(figsize=[16.5, 4.5])
    axlabfs, tiklabfs, splabfs = 12, 10, 14

    # set up ax1 and plot delPhi and event series there
    ax1 = fig.add_axes([0.1, 0.12, 0.85, 0.80])
    ax1.plot(ct, del_phi,
             c="Maroon", zorder=5,
             )
    ylo, yhi = ax1.get_ylim()
    ax1.bar(x=te,
            width=31 * np.ones(len(te)),
            height=(yhi - ylo) * np.ones(len(te)),
            bottom=ylo * np.ones(len(te)),
            edgecolor="none",
            facecolor="Turquoise",
            zorder=1,
            )

    # prettify ax1
    xlo, xhi = dt.datetime(1870, 1, 1), dt.datetime(2015, 12, 31)
    ax1.set_xlim(xlo, xhi)
    XMajorLocator = mdates.YearLocator(base=10, month=6, day=15)
    XMinorLocator = mdates.YearLocator(base=2, month=6, day=15)
    XMajorFormatter = mdates.DateFormatter("%Y")
    ax1.xaxis.set_major_locator(XMajorLocator)
    ax1.xaxis.set_minor_locator(XMinorLocator)
    ax1.xaxis.set_major_formatter(XMajorFormatter)
    ax1.set_ylim(ylo, yhi)
    ax1.grid(which="both")
    ax1.tick_params(which="major", size=8, direction="out")
    ax1.tick_params(which="minor", size=5, direction="out")
    ax1.tick_params(axis="both", labelsize=tiklabfs)
    ax1.set_xlabel("Time", fontsize=axlabfs)
    ax1.set_ylabel(r"$\Delta\phi = \phi_{ISMR} - \phi_{NINO}$",
                   fontsize=axlabfs)

    # save figure
    figname = "../plots/02_delphi_timeseries.png"
    pl.savefig(figname)
    print("figure saved to: %s" % figname)
    return None


def delphi_histogram(del_phi_dot, lothres, hithres):
    """
    Plots the histogram of instantaneous phase differences.
    """
    # set up figure
    fig = pl.figure(figsize=[6.5, 6.5])
    axlabfs, tiklabfs, splabfs = 12, 10, 14

    # plot histogram of derivative of del_phi
    ax1 = fig.add_axes([0.12, 0.12, 0.85, 0.85])
    h, be = np.histogram(del_phi_dot, bins="fd")
    bc = 0.5 * (be[1:] + be[:-1])
    ax1.fill_between(bc, h,
                     color="Maroon",
                     )
    ax1.fill_between(bc, h,
                     color="Turquoise",
                     where=(bc >= lothres) * (bc <= hithres),
                     )

    # show vertical lines to indicate the interval we choose for del_phi ~ 0
    ax1.axvline(lothres, color="k", linestyle="--")
    ax1.axvline(hithres, color="k", linestyle="--")

    # prettify ax1
    ax1.grid()
    ax1.set_xlabel(r"$\frac{\Delta\phi}{\mathrm{d}t}$",
                   fontsize=axlabfs)
    ax1.set_ylabel("Histogram counts", fontsize=axlabfs)
    ax1.tick_params(axis="both", labelsize=tiklabfs)
    _, yhi = ax1.get_ylim()
    ax1.set_ylim(0., yhi)

    # save figure
    figname = "../plots/03_delphi_histogram.png"
    pl.savefig(figname)
    print("figure saved to: %s" % figname)

    return None
