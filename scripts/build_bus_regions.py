# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Creates Voronoi shapes for each bus representing both onshore and offshore
regions.
Relevant Settings
-----------------
.. code:: yaml
    countries:
.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`toplevel_cf`
Inputs
------
- ``resources/country_shapes.geojson``: confer :ref:`shapes`
- ``resources/offshore_shapes.geojson``: confer :ref:`shapes`
- ``networks/base.nc``: confer :ref:`base`
Outputs
-------
- ``resources/regions_onshore.geojson``:
    .. image:: ../img/regions_onshore.png
        :scale: 33 %
- ``resources/regions_offshore.geojson``:
    .. image:: ../img/regions_offshore.png
        :scale: 33 %
Description
-----------
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import REGION_COLS, configure_logging
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, MultiPolygon

logger = logging.getLogger(__name__)


def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the
    polygon `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py
    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    points = np.array(points)

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 100, ymin - 100],
                        [xmin - 100, ymax + 100],
                        [xmax + 100, ymin - 100],
                        [xmax + 100, ymax + 100],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)
            poly = poly.intersection(outline)

            polygons.append(poly)

    return polygons

def assign_offshore_region(x, y, offshore_regions):
    region = offshore_regions.loc[offshore_regions.contains(Point(x, y))].index
    if not region.empty:
        return region[0]
    offshore_regions['distance'] = offshore_regions.apply(lambda reg: gpd.GeoSeries(Point(x,y)).distance(reg.geometry, align=False), axis=1)
    region = offshore_regions['distance'].idxmin()
    return region

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_bus_regions")
    configure_logging(snakemake)

    countries = snakemake.config["countries"]

    n = pypsa.Network(snakemake.input.base_network)

    build_custom_busmap = snakemake.config["enable"].get("build_custom_busmap", False)
    if build_custom_busmap:
        busmap = pd.read_csv('data/custom_busmap_sskern.csv')
        busmap['name'] = busmap['name'].astype(str)

    offshore_busmap = pd.DataFrame([])

    country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index("name")[
        "geometry"
    ]
    offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes)
    offshore_shapes = offshore_shapes.reindex(columns=REGION_COLS).set_index("name")[
        "geometry"
    ]
    meshed_offshore_shapes = gpd.read_file(snakemake.input.meshed_offshore_shapes)
    meshed_offshore_shapes = meshed_offshore_shapes.reindex(columns=REGION_COLS).set_index("name")

    onshore_regions = []
    offshore_regions = []

    for country in countries:
        c_b = n.buses.country == country

        onshore_shape = country_shapes[country]
        onshore_locs = n.buses.loc[c_b & n.buses.substation_lv, ["x", "y"]]
        onshore_regions.append(
            gpd.GeoDataFrame(
                {
                    "name": onshore_locs.index,
                    "x": onshore_locs["x"],
                    "y": onshore_locs["y"],
                    "geometry": voronoi_partition_pts(
                        onshore_locs.values, onshore_shape
                    ),
                    "country": country,
                }
            )
        )

    onshore_regions = pd.concat(onshore_regions, ignore_index=True)
    onshore_regions.to_file(
        snakemake.output.regions_onshore
    )

    for country in countries:
        meshed_shapes = meshed_offshore_shapes.loc[meshed_offshore_shapes.country == country]
        c_b = n.buses.country == country
        offshore_locs = n.buses.loc[c_b & n.buses.substation_off, ["x", "y"]]
        offshore_region_bus = offshore_locs.apply(lambda bus: assign_offshore_region(bus.x, bus.y, meshed_shapes), axis=1)
        for index_shape, offshore_shape in meshed_shapes.iterrows():
            r_b = offshore_region_bus == index_shape
            offshore_locs = n.buses.loc[offshore_region_bus.loc[r_b].index, ["x", "y"]]
            n.madd(
                "Bus",
                names=[offshore_shape.name],
                v_nom=220,
                x=offshore_shape.x,
                y=offshore_shape.y,
                substation_lv=False,
                substation_off=True,
                country=offshore_shape.country,
            )
            if offshore_locs.empty:
                offshore_regions_c = gpd.GeoDataFrame(
                    {
                        "name": offshore_shape.name,
                        "x": offshore_shape.x,
                        "y": offshore_shape.y,
                        "geometry": [offshore_shape.geometry],
                        "country": offshore_shape.country
                    }, index = pd.Index([offshore_shape.name], dtype='object', name='Bus')
                )
                offshore_regions.append(offshore_regions_c)

            else:
                offshore_regions_c = gpd.GeoDataFrame(
                    {
                        "name": offshore_locs.index,
                        "x": offshore_locs["x"],
                        "y": offshore_locs["y"],
                        "geometry": voronoi_partition_pts(offshore_locs.values, offshore_shape.geometry),
                        "country": offshore_shape.country,
                    }
                )
                offshore_regions_c = offshore_regions_c[offshore_regions_c.geometry.is_empty == False]
                offshore_regions.append(offshore_regions_c)
            index_shape_rest = index_shape.split('_')[2] == '0'
            if build_custom_busmap and not index_shape_rest:
                onshore_regions_country = onshore_regions.loc[onshore_regions.country==country].name.values
                add_offshore_busmap = pd.DataFrame(offshore_regions_c.name)
                add_offshore_busmap = add_offshore_busmap.assign(busmap=index_shape)
                offshore_busmap = pd.concat([offshore_busmap, add_offshore_busmap], ignore_index=True)
                if index_shape not in busmap.name.values:
                    add_to_busmap = pd.DataFrame({'name': [index_shape], 'busmap': [index_shape]})
                    busmap = pd.concat([busmap, add_to_busmap], ignore_index=True)

    if build_custom_busmap:
        busmap.to_csv(snakemake.output.busmap, index=False)
        offshore_busmap.to_csv(snakemake.output.busmap_offshore, index=False)

    if offshore_regions:
        pd.concat(offshore_regions, ignore_index=True).to_file(
            snakemake.output.regions_offshore
        )
    else:
        offshore_shapes.to_frame().to_file(snakemake.output.regions_offshore)

    n.export_to_netcdf(snakemake.output.network)
