# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
import logging
import os

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import pandas as pd
import pyomo.environ as po
import pypsa
from _helpers import configure_logging
from add_electricity import load_costs
from geopy.distance import geodesic
from scipy.spatial import Voronoi
from shapely.geometry import LineString
from sklearn.neighbors import BallTree
from spopt.region import Spenc
from spopt.region.maxp import MaxPHeuristic

logger = logging.getLogger(__name__)

def move_generators(offshore_regions, cluster_map):
    offshore_regions = gpd.read_file(offshore_regions)
    cluster_map = pd.read_csv(cluster_map).set_index('name')
    move_generators = (
        n.generators[n.generators.bus.isin(offshore_regions.name.unique())]
        .filter(like="offwind", axis=0)
        .index.to_series()
        .str.replace(" offwind-\w+", "", regex=True)
    ).rename('name')
    
    move_generators = pd.merge(move_generators, cluster_map, left_on='name', right_index=True)

    move_generators = move_generators[move_generators.isin(n.buses.index)]
    n.generators.loc[move_generators.index, "bus"] = move_generators.busmap
    # only consider turbine cost and substation cost for offshore generators connected to offshore grid
    n.generators.loc[move_generators.index, "capital_cost"] = (
        n.generators.loc[move_generators.index, "turbine_cost"]
        + costs.at["offwind-ac-station", "capital_cost"]
    )

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_offshore_grid", simpl="", clusters="60", offgrid="all"
        )
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.network)

    onshore_regions = gpd.read_file(snakemake.input.onshore_regions)

    offshore_regions = gpd.read_file(snakemake.input.offshore_regions)

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )

    move_generators(snakemake.input.offshore_regions, snakemake.input.busmap_offshore)

    add_offshore_connections()

    n.export_to_netcdf(snakemake.output[0])