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

geo_opts  = dict(min_height=400, cmap=bmy, global_extent=False, logz=True, colorbar=True, responsive=True)
elev_opts = dict(min_height=400, show_grid=True, responsive=True)
temp_opts = dict(min_height=400, fill_color='#f1948a', default_tools=[], toolbar=None, alpha=1.0, responsive=True)
prcp_opts = dict(min_height=400, fill_color='#85c1e9', default_tools=[], toolbar=None, alpha=1.0, responsive=True)

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

# tricks to workaround https://github.com/ioam/holoviews/issues/2730
def set_active_drag_raster(plot, element):
    plot.state.toolbar.active_drag = plot.state.tools[2]
def set_active_drag_shade(plot, element):
    plot.state.toolbar.active_drag = plot.state.tools[0]

static_geo  = rasterize(geo(data),   **geo_kw).options(alpha=0.1, tools=['hover', 'box_select'], finalize_hooks=[set_active_drag_raster], **geo_opts)
static_elev = datashade(elev(data), **elev_kw).options(alpha=0.1, tools=[         'box_select'], finalize_hooks=[set_active_drag_shade], **elev_opts)
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

    dyn_geo     = rasterize(Dynamic(dyn_data, operation=geo),   **geo_kw).options( **geo_opts)
    dyn_elev    = datashade(Dynamic(dyn_data, operation=elev), **elev_kw).options(**elev_opts)
    dyn_temp    =           Dynamic(dyn_data, operation=temp)
    dyn_prcp    =           Dynamic(dyn_data, operation=prcp)
    dyn_count   =           Dynamic(dyn_data, operation=count)

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

    title       = '<p style="font-size:35px">World glaciers explorer</p>'
    instruction = 'Box-select on each plot to subselect; clear selection to reset.<br>' + \
                  'See the <a href="https://github.com/panel-demos/glaciers">Jupyter notebook</a> source code for how to build apps like this!'
    oggm_logo   = '<a href="https://oggm.org"><img src="https://raw.githubusercontent.com/OGGM/oggm/master/docs/_static/logos/oggm_s_alpha.png" width=170></a>'
    pv_logo     = '<a href="https://pyviz.org"><img src="http://pyviz.org/assets/PyViz_logo_wm.png" width=80></a>'

    header = pn.Row(pn.Pane(oggm_logo), pn.layout.Spacer(width=30),
                    pn.Column(pn.Pane(title, height=25, width=400), pn.Spacer(height=-15), pn.Pane(instruction, width=500)),
                    pn.layout.HSpacer(), pn.Column(pn.Pane(dyn_count), pn.layout.Spacer(height=20), clear_button),
                    pn.Pane(pv_logo, width=80))

    return pn.Column(header, pn.Row(geomap, elevation), pn.Row(temperature, precipitation))

panel = get_oggm_panel()
panel.servable()
