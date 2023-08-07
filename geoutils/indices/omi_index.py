# %%
import xarray as xr
import pandas as pd
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.statistic_utils as sut
import geoutils.indices.indices_utils as iut
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as cplt
from importlib import reload


if __name__ == '__main__':
    reload(tu)
    reload(gut)
    folder = '/home/strnad/data/omi/'
    filename = 'omi.1x.txt'
    fname = f'{folder}/{filename}'

    header = ['Year', 'Month', 'Day', 'Hour', 'EOF1', 'EOF2', 'BSISO-ampl']
    df = pd.read_csv(fname,
                     #  skiprows=1,
                     header=None,
                     names=header,
                     delim_whitespace=True
                     )
    data_eof1 = df[['EOF1']].to_numpy().flatten()
    data_eof2 = df[['EOF2']].to_numpy().flatten()
    BSISO_ampl = df[['BSISO-ampl']].to_numpy().flatten()

    # %%
    dates = tu.get_dates_of_time_range(['1979-01-01', '2022-12-31'], freq='D')
    eof1 = xr.DataArray(data=data_eof1, name='BSISO1',
                        coords={"time": dates},
                        dims=["time"])
    eof2 = xr.DataArray(data=data_eof2, name='BSISO2',
                        coords={"time": dates},
                        dims=["time"])
    BSISO_ampl = xr.DataArray(data=BSISO_ampl, name='BSISO-ampl',
                              coords={"time": dates},
                              dims=["time"])

    angle_1 = np.rad2deg(np.arctan2(
        # eof2, eof1
        sut.standardize(eof2),  # X-axis is PC2
        sut.standardize(eof1)  # Y-axis is PC1
    ))
    # angle_1 = np.rad2deg(np.arctan2(data['BSISO1-2'],
    #                                 data['BSISO1-1']
    #                                 ))
    phase_1 = iut.get_phase_of_angle(angle_1)
    phase_1 = xr.DataArray(data=phase_1, name='BSISO-phase',
                           coords={"time": dates},
                           dims=["time"])
    # %%
    omi = xr.merge([eof1, eof2, BSISO_ampl, phase_1])

    savepath_omi = f'{folder}/omi.nc'
    fut.save_ds(omi, savepath_omi)

    # %%
    # check for consistency
    reload(bds)
    reload(cplt)
    data_dir = "/home/strnad/data/"

    dataset_file = data_dir + \
        f"climate_data/2.5/era5_ttr_{2.5}_ds.nc"

    ds_olr_25 = bds.BaseDataset(data_nc=dataset_file,
                                can=True,
                                an_types=['dayofyear', 'month', 'JJAS'],
                                )
    # %%
    import geoutils.indices.bsiso_index as bs
    reload(bs)
    nrows = 2
    ncols = 4
    im = cplt.create_multi_plot(nrows=nrows,
                                ncols=ncols,
                                projection='PlateCarree',
                                hspace=0.45, wspace=0.15,
                                orientation='horizontal',
                                lon_range=[-30, 180],
                                lat_range=[-30, 60])

    index_def = 'Kiladis'
    for idx, phase in enumerate(np.arange(1, 9)):
        tps = bs.get_bsisophase_tps(
            phase_number=phase,
            start_month='Jun',
            end_month='Sep',
            active=True,
            index_def=index_def)

        composite_pr = tu.get_sel_tps_ds(ds=ds_olr_25.ds, tps=tps)
        mean_pr = composite_pr.mean(dim='time')

        an_type = 'month'
        var_type = f'an_{an_type}'
        # var_type = 'ttr'
        vmax = 20
        vmin = -vmax

        im_comp = cplt.plot_map(mean_pr[var_type],
                                ax=im['ax'][idx],
                                plot_type='contourf',
                                cmap='RdBu_r',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                title=f"BSISO Phase {phase} ({len(tps)} days)",
                                label=rf'Anomalies OLR (wrt {an_type}) [W/m$^2$]',
                                )
    plot_dir = "/home/strnad/data/plots/bsiso/"
    savepath = plot_dir + \
        f"definitions/bsiso_phase_{index_def}_olr_{an_type}.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])
