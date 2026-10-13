"""Four seasons in one figure: monthly means sharing one colour scale.

Run:  python examples/03_multipanel_shared_colorbar.py
"""
import geoutils.plotting as gp
from common import FIGURES, label, load

res = gp.create_multi_plot(2, 2, projection="PlateCarree", figsize=(13, 9),
                           title="Monthly mean 2 m temperature, 2024")
names = {1: "January", 4: "April", 7: "July", 10: "October"}
for ax, (month, name) in zip(res["ax"], names.items()):
    t2m = load("t2m", month)
    im = gp.plot_map(t2m.mean("time").compute(), ax=ax, plot_type="contourf",
                     levels=12, title=name, vmin=-20, vmax=40, cmap="Spectral_r")
gp.add_colorbar(im, res["fig"], label(t2m), multi_plots=True)

gp.save_fig(FIGURES / "03_multipanel_shared_colorbar", fig=res["fig"])
