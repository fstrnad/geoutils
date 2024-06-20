# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.statistic_utils as sut
import geoutils.indices.indices_utils as iut
import numpy as np
import geoutils.geodata.base_dataset as bds

import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload
import pandas as pd

data_dir = "/home/strnad/data/"
plot_dir = "/home/strnad/data/plots/enso/"


def get_mei_index(start_month='Jan', end_month='Dec'):
    """Wrapper for get_meiv2_index

    Args:
        start_month (str, optional): start month. Defaults to 'Jan'.
        end_month (str, optional): end month. Defaults to 'Dec'.

    Returns:
        xr.dataarray: dataarray with MEI index
    """
    return get_meiv2_index(start_month=start_month, end_month=end_month)


def get_meiv2_index(start_month='Jan', end_month='Dec'):
    savepath_meiv2 = '/home/strnad/data/meiv2/meiv2_days.nc'
    mei_index = xr.open_dataset(savepath_meiv2)['MEIV2']
    mei_index = tu.get_month_range_data(mei_index,
                                        start_month=start_month,
                                        end_month=end_month,
                                        verbose=False)
    return mei_index


def get_enso_types(start_month='Jan', end_month='Dec'):
    mei_index = get_meiv2_index(start_month=start_month, end_month=end_month)
    # All ENSO time points
    nino_tps = mei_index[mei_index >= 0.5]
    nina_tps = mei_index[mei_index <= -0.5]
    neutral_tps = mei_index[(mei_index < 0.5) & (mei_index > -0.5)]
    if len(nino_tps) + len(nina_tps) + len(neutral_tps) != len(mei_index):
        raise ValueError('Some time points are missing')
    enso_types = {
        'El Nino': nino_tps,
        'Neutral': neutral_tps,
        'La Nina': nina_tps,
    }
    return enso_types


def get_enso_tps(start_month='Jan', end_month='Dec', enso_type='nino'):
    enso_types = get_enso_types(start_month=start_month, end_month=end_month)

    if enso_type == 'nino':
        enso_tps = enso_types['El Nino']
    elif enso_type == 'nina':
        enso_tps = enso_types['La Nina']
    elif enso_type == 'neutral':
        enso_tps = enso_types['neutral']
    else:
        raise ValueError(f'Unknown enso_type: {enso_type}')

    return enso_tps


if __name__ == '__main__':

    # Only available as monthly data
    mei_file = '/home/strnad/data/meiv2/meiv2_2023.data'

    data = pd.read_csv(mei_file, delim_whitespace=True,
                       skiprows=1,
                       names=tu.months
                       )

    reload(tu)
    tps_mei_months = tu.get_dates_of_time_range(
        time_range=['1979-01', '2023-12'],
        freq='M')  # Downloaded 05.02.2024

    only_data = data.to_numpy().flatten()
    # %%
    meiv2_index = xr.DataArray(
        data=only_data,
        dims=['time'],
        coords=dict(
            time=tps_mei_months
        ),
        name='MEIV2'
    )
    savepath_meiv2 = '/home/strnad/data/meiv2/meiv2.nc'
    meiv2_index.to_netcdf(savepath_meiv2)
    # %%
    # Interpolate to days
    tps_mei_days = tu.get_dates_of_time_range(
        time_range=['1979-01', '2023-12'],
        freq='D')
    meiv2_index_days = meiv2_index.interp(time=tps_mei_days)

    savepath_meiv2 = '/home/strnad/data/meiv2/meiv2_days.nc'
    meiv2_index_days.to_netcdf(savepath_meiv2)

    # %%
    reload(cplt)
    meiv2_index_days = get_meiv2_index()
    enso_types = get_enso_types()
    nino_tps = enso_types['El Nino']
    nina_tps = enso_types['La Nina']
    neutral_tps = enso_types['Neutral']
    im = cplt.plot_xy(x_arr=[meiv2_index_days.time,
                             nino_tps.time, nina_tps.time, neutral_tps.time],
                      y_arr=[meiv2_index_days,
                             nino_tps, nina_tps, neutral_tps],
                      label_arr=['MEIv2',
                                 'El Nino', 'La Nina', 'Neutral'],)
    cplt.plot_hline(ax=im['ax'], y=0.5, color='k', linestyle='--',
                    lw=3)
    cplt.plot_hline(ax=im['ax'], y=-0.5, color='k', linestyle='--',
                    lw=3)

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
    tps_arr = [
        nino_tps,
        # neutral_tps,
        nina_tps,
    ]
    groups = ['El Nino', 'Neutral', 'La Nina']
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
        f"definitions/sst_{an_type}_enso_types.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])

# %%