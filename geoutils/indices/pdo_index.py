# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload
import pandas as pd
import geoutils.geodata.base_dataset as bds

data_dir = "/home/strnad/data/"
plot_dir = "/home/strnad/data/plots/pdo/"


def get_pdo_index(start_month='Jan', end_month='Dec',
                  time_range=None):
    savepath_pdo = '/home/strnad/data/pdo/pdo_days.nc'
    pdo_index = xr.open_dataset(savepath_pdo)['pdo']
    pdo_index = tu.get_month_range_data(pdo_index,
                                        start_month=start_month,
                                        end_month=end_month,
                                        verbose=False)
    if time_range is not None:
        pdo_index = tu.get_time_range_data(pdo_index,
                                           time_range=time_range,
                                           verbose=False)
    return pdo_index


def get_pdo_types(start_month='Jan', end_month='Dec',
                  time_range=None,
                  threshold=0):
    pdo_index = get_pdo_index(start_month=start_month,
                              end_month=end_month,
                              time_range=time_range)
    # All ENSO time points
    pos_tps = pdo_index[pdo_index >= np.abs(threshold)]
    neg_tps = pdo_index[pdo_index <= -1*np.abs(threshold)]
    if threshold != 0:
        pos_tps = pdo_index[pdo_index >= np.abs(threshold)]
        neg_tps = pdo_index[pdo_index <= -1*np.abs(threshold)]
        neutral_tps = pdo_index[np.abs(pdo_index) < threshold]
    else:
        pos_tps = pdo_index[pdo_index > np.abs(threshold)]
        neg_tps = pdo_index[pdo_index <= -1*np.abs(threshold)]
        neutral_tps = []
    if len(pos_tps) + len(neg_tps) + len(neutral_tps) != len(pdo_index):
        gut.myprint(f'pos: {len(pos_tps)}, neg: {len(neg_tps)}, neutral: {len(neutral_tps)}, PDO index: {len(pdo_index)}')
        raise ValueError('Some time points are missing')
    pdo_types = {
        'positive': pos_tps,
        'negative': neg_tps,
    }
    if threshold != 0:
        pdo_types['neutral'] = neutral_tps
    return pdo_types


if __name__ == '__main__':

    # Only available as monthly data
    # Downloaded from NOAA: https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat
    pdo_file = '/home/strnad/data/pdo/ersst.v5.pdo.dat'

    data = pd.read_csv(pdo_file, delim_whitespace=True,
                       skiprows=1,
                       names=tu.months
                       )

    reload(tu)
    tps_pdo_months = tu.get_dates_of_time_range(
        time_range=['1854-01', '2023-12'],
        freq='M')  # Downloaded 05.02.2024

    only_data = data.to_numpy().flatten()
    # %%
    pdo_index = xr.DataArray(
        data=only_data,
        dims=['time'],
        coords=dict(
            time=tps_pdo_months
        ),
        name='pdo'
    )
    savepath_pdo = '/home/strnad/data/pdo/pdo_index.nc'
    pdo_index.to_netcdf(savepath_pdo)
    # %%
    # Interpolate to days
    tps_pdo_days = tu.get_dates_of_time_range(
        time_range=['1854-01', '2023-12'],
        freq='D')
    pdo_index_days = pdo_index.interp(time=tps_pdo_days)

    savepath_pdo = '/home/strnad/data/pdo/pdo_days.nc'
    pdo_index_days.to_netcdf(savepath_pdo)

    # %%
    reload(cplt)
    pdo_types = get_pdo_types()
    pos_tps = pdo_types['positive']
    neg_tps = pdo_types['negative']
    cplt.plot_xy(x_arr=[pos_tps.time, neg_tps.time],
                 y_arr=[pos_tps, neg_tps],
                 label_arr=['Pos PDO', 'Neg PDO'],
                 xlabel='Time',
                 ylabel='PDO Index',)

    # %%
    # Plot the different PDO types
    reload(cplt)
    dataset_file = data_dir + \
        f"/climate_data/2.5/era5_sst_{2.5}_ds.nc"

    ds_sst = bds.BaseDataset(data_nc=dataset_file,
                             can=True,
                             an_types=['JJAS', 'month'],
                             month_range=['Jun', 'Sep'],
                             #  lon_range=lon_range_cut,
                             #  lat_range=lat_range_cut,
                             )
    # %%
    pdo_types = get_pdo_types(start_month='Jun',
                              end_month='Sep',
                              time_range=['1980-01', '2020-12'],
                              threshold=0.5)
    pos_tps = pdo_types['positive']
    neg_tps = pdo_types['negative']
    tps_arr = [
        neg_tps,
        pos_tps,
    ]
    groups = ['Negative PDO', 'Positive PDO', ]
    an_type = 'month'
    var_type = f'sst_an_{an_type}'
    label_sst = f'SST Anomalies (wrt {an_type}) [K]'
    vmin_sst = -1
    vmax_sst = -vmin_sst

    im = cplt.create_multi_plot(nrows=1,
                                ncols=len(tps_arr),
                                orientation='horizontal',
                                # hspace=0.7,
                                wspace=0.2,
                                projection='PlateCarree',
                                lat_range=[-50, 70],
                                lon_range=[30, -60],
                                dateline=True,
                                )

    for idx, sel_tps in enumerate(tps_arr):
        group = groups[idx]
        mean, mask = tu.get_mean_tps(ds_sst.ds[var_type], tps=sel_tps)

        im_sst = cplt.plot_map(mean*mask,
                               ax=im['ax'][idx],
                               title=f'{group} (JJAS)',
                               cmap='RdBu_r',
                               plot_type='contourf',
                               levels=14,
                               centercolor='white',
                               vmin=vmin_sst, vmax=vmax_sst,
                               extend='both',
                               orientation='horizontal',
                            #    significance_mask=mask,
                               hatch_type='..',
                               label=label_sst,
                               land_ocean=True,
                               )
    savepath = plot_dir + \
        f"definitions/sst_{an_type}_pdo_types.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])
