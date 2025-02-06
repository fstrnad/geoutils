from collections import OrderedDict
import pandas as pd
import pgeocode
import os
import zipfile
import requests


def download_file(url, local_filename):
    # variant of http://stackoverflow.com/a/16696317
    if not os.path.exists(local_filename):
        r = requests.get(url, stream=True)
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    return local_filename


def get_anlagenregister():
    eeg_fn = download_file(
        "http://www.energymap.info/download/eeg_anlagenregister_2015.08.utf8.csv.zip",
        "eeg_anlagenregister_2015.08.utf8.csv.zip",
    )

    with zipfile.ZipFile(eeg_fn, "r") as zip_ref:
        zip_ref.extract("eeg_anlagenregister_2015.08.utf8.csv")


def load_capacities(typ, cap_range=None, until=None):
    """
    Read in and select capacities.

    Parameters
    ----------
        typ : str
            Type of energy source, e.g. "Solarstrom" (PV), "Windenergie" (wind).
        cap_range : (optional) list-like
            Two entries, limiting the lower and upper range of capacities (in kW)
            to include. Left-inclusive, right-exclusive.
        until : str
            String representation of a datetime object understood by pandas.to_datetime()
            for limiting to installations existing until this datetime.

    """
    # Load locations of installed capacities and remove incomplete entries
    cols = OrderedDict(
        (
            ("installation_date", 0),
            ("plz", 2),
            ("city", 3),
            ("type", 6),
            ("capacity", 8),
            ("level", 9),
            ("lat", 19),
            ("lon", 20),
            ("validation", 22),
        )
    )
    database = pd.read_csv(
        "eeg_anlagenregister_2015.08.utf8.csv",
        sep=";",
        decimal=",",
        thousands=".",
        comment="#",
        header=None,
        usecols=list(cols.values()),
        names=list(cols.keys()),
        # German postal codes can start with '0' so we need to treat them as str
        dtype={"plz": str},
        parse_dates=["installation_date"],
        date_format="%d.%m.%Y",
        na_values=("O04WF", "keine"),
    )

    database = database[(database["validation"] == "OK")
                        & (database["plz"].notna())]

    # Query postal codes <-> coordinates mapping
    de_nomi = pgeocode.Nominatim("de")
    plz_coords = de_nomi.query_postal_code(database["plz"].unique())
    plz_coords = plz_coords.set_index("postal_code")

    # Fill missing lat / lon using postal codes entries
    database.loc[database["lat"].isna(), "lat"] = database["plz"].map(
        plz_coords["latitude"]
    )
    database.loc[database["lon"].isna(), "lon"] = database["plz"].map(
        plz_coords["longitude"]
    )

    # Ignore all locations which have not be determined yet
    database = database[database["lat"].notna() & database["lon"].notna()]

    # Select data based on type (i.e. solar/PV, wind, ...)
    data = database[database["type"] == typ].copy()

    # Optional: Select based on installation day
    if until is not None:
        data = data[data["installation_date"] < pd.to_datetime(until)]

    # Optional: Only installations within this caprange (left inclusive, right exclusive)
    if cap_range is not None:
        data = data[
            (cap_range[0] <= data["capacity"]) & (
                data["capacity"] < cap_range[1])
        ]

    data["capacity"] = data.capacity / 1e3  # convert to MW
    return data.rename(columns={"lon": "x", "lat": "y"})
