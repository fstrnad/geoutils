import os
import xarray as xr
from tabnanny import check
import geopandas as gpd
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

not_europe_countries = ['Russia',
                        'Belarus',
                        'Jersey',
                        'Isle of Man',
                        # 'Iceland',
                        'Faroe Islands',
                        'Guernsey'
                        ]


country_lon_lat_ranges = {
    'Germany': {'lon': [3, 18.75], 'lat':  [43, 58.75]},
    'Britain': {'lon': [-12, 3.75], 'lat': [46.5, 62.25]}
}


def get_country_lon_lat_range(country_name):
    """
    Function to get the longitude and latitude range of a country.

    Parameters:
    country_name (str): The name of the country.

    Returns:
    dict: A dictionary with 'lon' and 'lat' keys containing the respective ranges.
    """
    if country_name in country_lon_lat_ranges:
        return country_lon_lat_ranges[country_name]
    else:
        raise ValueError(f"Longitude and latitude range for {country_name} is not defined.")

def get_country_file():
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    return countries


def list_all_countries():
    countries = get_country_file()
    country_names = [r.attributes["NAME_EN"] for r in countries]
    return country_names


def check_country_exist(country_name):
    all_countries = list_all_countries()
    if country_name in all_countries:
        return True
    else:
        raise ValueError(f"{country_name} is not a valid country name.")


def get_countries_offshore():
    import country_converter as coco
    file = os.path.abspath(__file__)
    wd = os.path.dirname(file)
    offshore_countries = gpd.read_file(f'{wd}/offshore_shapes.geojson')
    cc = coco.CountryConverter()
    country_names = list(offshore_countries.name)
    countries_en = cc.convert(names=country_names, to='name_short')
    offshore_countries['name'] = countries_en

    return offshore_countries


def get_countries(onshore=True):
    if onshore:
        countries = get_country_file()
        country_shapes = gpd.GeoSeries(
            {r.attributes["NAME_EN"]: r.geometry for r in countries},
            crs={"init": "epsg:4326"},
        )
    else:
        countries = get_countries_offshore()
        country_shapes = gpd.GeoSeries(
            {r.name: r.geometry for r in countries.itertuples()},
            crs={"init": "epsg:4326"},
        )

    return country_shapes


def get_countries_in_continent(continent_name):
    """
    Function to get all countries in a specific continent using Cartopy's shapefile data.

    Parameters:
    continent_name (str): The name of the continent to filter by (e.g., 'Europe', 'Asia', etc.).

    Returns:
    List of country names belonging to the specified continent.
    """
    # Path to the shapefile (automatically handled by Cartopy)
    countries = get_country_file()

    # Initialize a list to store the country names
    country_names = []

    # Iterate over the records and filter by continent
    for record in countries:
        # Access the continent name from the attribute dictionary
        continent = record.attributes['CONTINENT']

        # If the country is in the specified continent, add its name to the list
        if continent == continent_name:
            country_names.append(record.attributes['NAME_EN'])

    if continent_name == 'Europe':
        country_names = [
            country for country in country_names if country not in not_europe_countries]

    return country_names


def get_country_table():
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )

    df = gpd.read_file(shpfilename)

    # set index to country name
    df = df.set_index('NAME_EN')

    return df


def get_country(country_names):
    countries = get_country_table()
    if isinstance(country_names, str):
        country_names = [country_names]
    for country in country_names:
        check_country_exist(country)
    return countries.loc[country_names]


def get_country_shape(country_names, onshore=True):
    if isinstance(country_names, str):
        country_names = [country_names]
    for country in country_names:
        check_country_exist(country)

    countries = get_countries(onshore=onshore)
    return countries.reindex(country_names)


def get_country_bounds(country_name):
    country_shape = get_country_shape(country_name)
    return country_shape.total_bounds


def cutout_country_cells(cutout, country_name, onshore=True, as_xr=True):
    country_shape = get_country_shape(country_name, onshore=onshore)
    indicator_matrix = cutout.indicatormatrix(country_shape)
    if as_xr:
        indicator_matrix = xr.DataArray(
            indicator_matrix.toarray().reshape(cutout.shape),
            dims=["lat", "lon"],
            coords=[cutout.coords["lat"], cutout.coords["lon"]],
        )

    return indicator_matrix


def get_country_iso(country_name):
    countries = get_country_table()
    return countries.loc[country_name].ISO_A2
