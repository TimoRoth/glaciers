import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import holoviews as hv
import datashader as ds
import geoviews as gv
import panel as pn

from colorcet import b_linear_bmy_10_95_c78 as bmy
from holoviews.util import Dynamic
from holoviews.operation.datashader import rasterize, datashade
from holoviews.streams import Stream, BoundsXY, BoundsX

hv.extension('bokeh', width=100)
pn.extension()

df = pd.read_csv('data/oggm_glacier_explorer.csv')
df['latdeg'] = df.cenlat

df.tail()

data = gv.Points(df, [('cenlon', 'Longitude'), ('cenlat', 'Latitude')],
                     [('avg_prcp', 'Annual Precipitation (mm/yr)'),
                      ('area_km2', 'Area'), ('latdeg', 'Latitude (deg)'),
                      ('avg_temp_at_mean_elev', 'Annual Temperature at avg. altitude'),
                      ('mean_elev', 'Elevation')])
total_area = df.area_km2.sum()

data = gv.operation.project_points(data).map(gv.Dataset, gv.Points).clone(crs=ccrs.GOOGLE_MERCATOR)

geo_kw    = dict(aggregator=ds.sum('area_km2'), x_sampling=1000, y_sampling=1000)
elev_kw   = dict(cmap='#7d3c98')
temp_kw   = dict(num_bins=50, adjoin=False, normed=False, bin_range=data.range('avg_temp_at_mean_elev'))
prcp_kw   = dict(num_bins=50, adjoin=False, normed=False, bin_range=data.range('avg_prcp'))

size_opts = dict(min_height=400, min_width=600, responsive=True)
geo_opts  = dict(size_opts, cmap=bmy, global_extent=False, logz=True, colorbar=True)
elev_opts = dict(size_opts, show_grid=True)
temp_opts = dict(size_opts, fill_color='#f1948a', default_tools=[], toolbar=None, alpha=1.0)
prcp_opts = dict(size_opts, fill_color='#85c1e9', default_tools=[], toolbar=None, alpha=1.0)

geo_bg = gv.tile_sources.EsriImagery.options(alpha=0.6, bgcolor="black")
geopoints = data.to(gv.Points, ['cenlon', 'cenlat'], ['area_km2'], []).options(**geo_opts).redim.range(area_km2=(0, 3000))

(geo_bg*rasterize(geopoints, **geo_kw).options(**geo_opts) +
 datashade(data.to(hv.Scatter, 'mean_elev','latdeg', []), **elev_kw).options(**elev_opts) +
 data.hist('avg_temp_at_mean_elev', **temp_kw).options(**temp_opts) +
 data.hist('avg_prcp',              **prcp_kw).options(**prcp_opts)).cols(2)

def geo(data):
    return gv.Points(data, crs=ccrs.GOOGLE_MERCATOR).options(alpha=1)

def elev(data):
    return data.to(hv.Scatter, 'mean_elev', 'latdeg', [])

def temp(data):
    return data.hist('avg_temp_at_mean_elev', **temp_kw).options(**temp_opts)

def prcp(data):
    return data.hist('avg_prcp', **prcp_kw).options(**prcp_opts)

def count(data):
    return hv.Div('<p style="font-size:20px">Glaciers selected: {}'.format(len(data)) + "<br>" +
                  'Area: {:.0f} km² ({:.1f}%)</font>'.format(np.sum(data['area_km2']), np.sum(data['area_km2']) / total_area * 100)).options(height=40)

static_geo  = rasterize(geo(data),   **geo_kw).options(alpha=0.1, tools=['hover', 'box_select'], active_tools=['box_select'], **geo_opts)
static_elev = datashade(elev(data), **elev_kw).options(alpha=0.1, tools=[         'box_select'], active_tools=['box_select'], **elev_opts)
static_temp = temp(data).options(alpha=0.1)
static_prcp = prcp(data).options(alpha=0.1)

def combine_selections(**kwargs):
    """
    Combines selections on all available plots into a single selection by index.
    """
    if all(not v for v in kwargs.values()):
        return slice(None)
    selection = {}
    for key, bounds in kwargs.items():
        if bounds is None:
            continue
        elif len(bounds) == 2:
            selection[key] = bounds
        else:
            xbound, ybound = key.split('__')
            selection[xbound] = bounds[0], bounds[2]
            selection[ybound] = bounds[1], bounds[3]
    return sorted(set(data.select(**selection).data.index))

def select_data(**kwargs):
    return data.iloc[combine_selections(**kwargs)] if kwargs else data

def get_oggm_panel():
    geo_bounds  = BoundsXY(source=static_geo,  rename={'bounds':  'cenlon__cenlat'})
    elev_bounds = BoundsXY(source=static_elev, rename={'bounds':  'mean_elev__latdeg'})
    temp_bounds = BoundsX(source=static_temp, rename={'boundsx': 'avg_temp_at_mean_elev'})
    prcp_bounds = BoundsX(source=static_prcp, rename={'boundsx': 'avg_prcp'})

    selections  = [geo_bounds, elev_bounds, temp_bounds, prcp_bounds]
    dyn_data    = hv.DynamicMap(select_data, streams=selections)

    dyn_geo   = rasterize(dyn_data.apply(geo),  **geo_kw).options( **geo_opts)
    dyn_elev  = datashade(dyn_data.apply(elev), **elev_kw).options(**elev_opts)
    dyn_temp  =           dyn_data.apply(temp)
    dyn_prcp  =           dyn_data.apply(prcp)
    dyn_count =           dyn_data.apply(count)

    geomap = geo_bg * static_geo  * dyn_geo
    elevation       = static_elev * dyn_elev
    temperature     = static_temp * dyn_temp
    precipitation   = static_prcp * dyn_prcp

    def clear_selections(arg=None):
        geo_bounds.update(bounds=None)
        elev_bounds.update(bounds=None)
        temp_bounds.update(boundsx=None)
        prcp_bounds.update(boundsx=None)
        Stream.trigger(selections)

    clear_button = pn.widgets.Button(name='Clear selection')
    clear_button.param.watch(clear_selections, 'clicks');

    title       = '<div style="font-size:35px">World glaciers explorer</div>'
    instruction = 'Box-select on each plot to subselect; clear selection to reset.<br>' + \
                  'See the <a href="https://github.com/panel-demos/glaciers">Jupyter notebook</a> source code for how to build apps like this!'
    oggm_logo   = '<a href="https://oggm.org"><img src="https://raw.githubusercontent.com/OGGM/oggm/master/docs/_static/logos/oggm_s_alpha.png" width=170></a>'
    pn_logo     = '<a href="https://panel.pyviz.org"><img src="https://panel.pyviz.org/_static/logo_stacked.png" width=140></a>'

    header = pn.Row(pn.Pane(oggm_logo), pn.layout.Spacer(width=30),
                    pn.Column(pn.Pane(title, height=25, width=400), pn.Spacer(height=-15), pn.Pane(instruction, width=500)),
                    pn.layout.HSpacer(), pn.Column(pn.Pane(dyn_count), pn.layout.Spacer(height=20), clear_button),
                    pn.Pane(pn_logo, width=140))

    return pn.Column(header, pn.Row(geomap, elevation), pn.Row(temperature, precipitation), width_policy='max', height_policy='max')

panel = get_oggm_panel()
panel.servable()
