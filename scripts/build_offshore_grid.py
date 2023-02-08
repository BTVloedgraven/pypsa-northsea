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


def consense(x):
    v = x.iat[0]
    assert (
        x == v
    ).all() or x.isnull().all(), "In {} cluster {} the values of attribute {} do not agree:\n{}".format(
        component, x.name, attr, x
    )
    return v


def center_of_mass(df, groupby=None, weight=None):
    x = (
        df.groupby(groupby)
        .apply(lambda x: (x["x"] * x[weight]).sum() / x[weight].sum())
        .rename("x")
    )
    y = (
        df.groupby(groupby)
        .apply(lambda x: (x["y"] * x[weight]).sum() / x[weight].sum())
        .rename("y")
    )
    return pd.concat([x, y], axis=1)


def get_region_intersections(regions):
    intesections = dict()
    regions = regions.to_crs(3035).buffer(100)
    for idx, region in regions.iteritems():
        intesections.update({idx: regions[regions.intersects(region)].index.tolist()})
    return intesections


def move_generators(offshore_regions, cluster_map=None):
    move_generators = (
        n.generators[n.generators.bus.isin(offshore_regions.bus.unique())]
        .filter(like="offwind", axis=0)
        .index.to_series()
        .str.replace(" offwind-\w+", "", regex=True)
    )

    # remove some buses from the move series as they could be attached to the same onshore bus, but are not within the sea region
    if cluster_map is not None:
        move_generators = move_generators.map(cluster_map)
        prefix = "off_"
        move_generators = prefix + move_generators
    elif "hub" in offshore_regions.columns:
        move_generators = move_generators.map(
            offshore_regions.hub.astype("str")
        ).dropna()
        prefix = "hub_"
        move_generators = prefix + move_generators
    else:
        prefix = "off_"
        move_generators = prefix + move_generators

    move_generators = move_generators[move_generators.isin(n.buses.index)]
    n.generators.loc[move_generators.index, "bus"] = move_generators
    # only consider turbine cost and substation cost for offshore generators connected to offshore grid
    n.generators.loc[move_generators.index, "capital_cost"] = (
        n.generators.loc[move_generators.index, "turbine_cost"]
        + costs.at["offwind-ac-station", "capital_cost"]
    )
    rename_index = dict(zip(move_generators.index, prefix + move_generators.index))
    n.generators.rename(index=rename_index, inplace=True)
    n.generators_t.p_max_pu.rename(columns=rename_index, inplace=True)


def add_offshore_connections():
    # Create line for every offshore bus and connect it to onshore buses
    onshore_coords = n.buses.loc[offshore_regions.bus.unique(), ["x", "y"]]
    offshore_buses_coord = n.buses.loc[n.buses.index.str.contains("off"), ["x", "y"]]
    offshore_hub_coord = n.buses.loc[n.buses.index.str.contains("hub"), ["x", "y"]]

    if offshore_hub_coord.empty:
        tree = BallTree(
            np.radians(offshore_buses_coord), leaf_size=40, metric="haversine"
        )
        offshore_coords = offshore_buses_coord
    else:
        tree = BallTree(
            np.radians(offshore_hub_coord), leaf_size=40, metric="haversine"
        )
        offshore_coords = offshore_hub_coord

    coords = pd.concat([onshore_coords, offshore_coords])
    coords["xy"] = list(map(tuple, (coords[["x", "y"]]).values))

    # works better than with closest neighbors. maybe only create graph like this for offshore buses:
    cells, generators = libpysal.cg.voronoi_frames(
        offshore_coords.values, clip="convex hull"
    )
    delaunay = libpysal.weights.Rook.from_dataframe(cells)
    offshore_line_graph = delaunay.to_networkx()
    offshore_line_graph = nx.relabel_nodes(
        offshore_line_graph, dict(zip(offshore_line_graph, offshore_coords.index))
    )

    offshore_lines = nx.to_pandas_edgelist(offshore_line_graph)

    # remove lines which intersect with onshore shapes
    lines_filter = offshore_lines.apply(
        lambda x: LineString(
            [n.buses.loc[x.source, ["x", "y"]], n.buses.loc[x.target, ["x", "y"]]]
        ),
        axis=1,
    )
    onshore_shape = onshore_regions.unary_union
    lines_filter = lines_filter.apply(lambda x: x.intersects(onshore_shape))
    offshore_lines.drop(offshore_lines[lines_filter].index, inplace=True)

    _, ind = tree.query(np.radians(onshore_coords), k=1)
    # Build line graph to connect all offshore nodes and

    on_line_graph = nx.Graph()
    for i, bus in enumerate(onshore_coords.index):
        for j in range(ind.shape[1]):
            bus1 = offshore_coords.index[ind[i, j]]
            on_line_graph.add_edge(bus, bus1)

    onshore_lines = nx.to_pandas_edgelist(on_line_graph)

    lines_df = pd.concat([offshore_lines, onshore_lines], axis=0, ignore_index=True)

    lines_df = lines_df.rename(
        columns={"source": "bus0", "target": "bus1", "weight": "length"}
    ).astype({"bus0": "string", "bus1": "string", "length": "float"})

    lines_df.loc[:, "length"] = lines_df.apply(
        lambda x: geodesic(coords.loc[x.bus0, "xy"], coords.loc[x.bus1, "xy"]).km,
        axis=1,
    )
    lines_df.drop(lines_df.query("length==0").index, inplace=True)
    lines_df.index = "off_" + lines_df.index.astype("str")

    if offshore_hub_coord.empty:
        n.madd(
            "Line",
            names=lines_df.index,
            v_nom=220,
            bus0=lines_df["bus0"].values,
            bus1=lines_df["bus1"].values,
            length=lines_df["length"].values,
            type="149-AL1/24-ST1A 110.0",
        )
        # attach cable cost AC for offshore grid lines
        line_length_factor = snakemake.config["lines"]["length_factor"]
        cable_cost = n.lines.loc[lines_df.index, "length"].apply(
            lambda x: x
            * line_length_factor
            * costs.at["offwind-ac-connection-submarine", "capital_cost"]
        )
        n.lines.loc[lines_df.index, "capital_cost"] = cable_cost
    else:
        n.madd(
            "Link",
            names=lines_df.index,
            carrier="DC",
            bus0=lines_df["bus0"].values,
            bus1=lines_df["bus1"].values,
            length=lines_df["length"].values,
        )
        # attach cable cost DC for offshore grid lines
        line_length_factor = snakemake.config["lines"]["length_factor"]
        cable_cost = n.links.loc[lines_df.index, "length"].apply(
            lambda x: x
            * line_length_factor
            * costs.at["offwind-dc-connection-submarine", "capital_cost"]
        )
        n.links.loc[lines_df.index, "capital_cost"] = cable_cost


def distribute_offshore_cluster(n_clusters, cluster_countries_weights, solver_name):
    m = po.ConcreteModel()

    m.x = po.Var(
        list(cluster_countries_weights.index),
        bounds=(1, n_clusters),
        domain=po.Integers,
    )
    m.tot = po.Constraint(expr=(po.summation(m.x) == n_clusters))
    m.objective = po.Objective(
        expr=sum(
            (m.x[i] - cluster_countries_weights.loc[i] * n_clusters) ** 2
            for i in cluster_countries_weights.index
        ),
        sense=po.minimize,
    )
    opt = po.SolverFactory(solver_name)
    if not opt.has_capability("quadratic_objective"):
        logger.warning(
            f"The configured solver `{solver_name}` does not support quadratic objectives. Falling back to `ipopt`."
        )
        opt = po.SolverFactory("ipopt")

    results = opt.solve(m)
    assert (
        results["Solver"][0]["Status"] == "ok"
    ), f"Solver returned non-optimally: {results}"

    return (
        pd.Series(m.x.get_values(), index=cluster_countries_weights.index)
        .round()
        .astype(int)
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_offshore_grid", simpl="", clusters="60", offgrid="all"
        )
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.clustered_network)
    n_simpl = pypsa.Network(snakemake.input.simplified_network)

    offgrid = snakemake.wildcards["offgrid"]
    if not offgrid:
        n.export_to_netcdf(snakemake.output[0])
    else:
        country_shapes = gpd.read_file(snakemake.input.country_shapes).set_index(
            "name"
        )["geometry"]

        onshore_regions = gpd.read_file(snakemake.input.onshore_regions)

        offshore_regions = gpd.read_file(snakemake.input.offshore_regions)

        offshore_shapes = gpd.read_file(snakemake.input.offshore_shapes).set_index(
            "name"
        )["geometry"]

        Nyears = n.snapshot_weightings.objective.sum() / 8760.0

        costs = load_costs(
            snakemake.input.tech_costs,
            snakemake.config["costs"],
            snakemake.config["electricity"],
            Nyears,
        )

        offshore_generators = (
            n_simpl.generators.filter(regex="offwind", axis=0)
            .loc[:, ["p_nom_max", "bus"]]
            .copy()
        )
        offshore_generators["cf"] = (
            n_simpl.generators_t.p_max_pu.loc[:, offshore_generators.index]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            / 8760
        )
        offshore_generators["regions"] = offshore_generators.index.str.replace(
            " offwind-\w+", "", regex=True
        )
        busmap = gpd.read_file(snakemake.input.busmap_cluster).loc[:, ['name', 'busmap']].rename(columns={"name": "regions", "busmap":"bus"})
        offshore_generators = offshore_generators.groupby("regions").agg(
            {"p_nom_max": np.sum, "cf": np.mean}
        )
        offshore_generators = offshore_generators.merge(busmap, on='regions')

        # offshore regions have more shapes than offshore generators have, don't know why, maybe rerun build renewable and add electricity and check again
        offshore_regions = offshore_regions.merge(
            offshore_generators, right_on="regions", left_on="name"
        ).set_index("name")

        # calculate distance to offshore region
        coords = pd.DataFrame(index=offshore_regions.index)
        coords["onshore"] = list(
            map(tuple, (n.buses.loc[offshore_regions.bus, ["x", "y"]]).values)
        )
        coords["offshore"] = list(
            map(tuple, (offshore_regions[["x", "y"]]).values)
        )
        offshore_regions["distance"] = coords.apply(
            lambda x: geodesic(x.onshore, x.offshore).km, axis=1
        )

        # only build grid for buses in country list and/or in sea shape
        countries = snakemake.config["offshore_grid"]["countries"]
        offshore_regions = offshore_regions.loc[
            offshore_regions.country.str.contains("|".join(countries))
        ]

        if snakemake.config["offshore_grid"]["sea_region"]:
            sea_shape = gpd.read_file(snakemake.input.north_sea_shape)
            offshore_regions = offshore_regions[
                offshore_regions.intersects(sea_shape.geometry.unary_union)
            ]

        # TODO: culster for offshore hubs
        # TODO: think about threshold criterion
        offshore_regions = offshore_regions.query("distance>=50 & p_nom_max>1000")
        offshore_regions["yield"] = offshore_regions.eval("p_nom_max * cf")

        # cluster buses to simplify grid or to get hubs
        if "c" in offgrid:
            n_clusters = int(offgrid.split("-")[0])
            # distribute clusters according to energy yield and load of the connected onshore buses
            cluster_countries_weights = offshore_regions["yield"].groupby(
                offshore_regions.country
            ).sum().transform(lambda x: x / x.max()) + n.loads_t.p_set.sum()[
                offshore_regions["bus"].unique()
            ].groupby(
                n.buses.country
            ).sum().transform(
                lambda x: x / x.max()
            )
            # dropna() needed, because it happened that in the load weight entries of other countries than the regarded occurred
            cluster_countries_weights = cluster_countries_weights.transform(
                lambda x: x / x.sum()
            ).dropna()

            n_clusters_country = distribute_offshore_cluster(
                n_clusters, cluster_countries_weights, "gurobi"
            )
            for country, n_cluster in n_clusters_country.iteritems():
                regions = offshore_regions.query("country==@country")
                if n_clusters == 1:
                    offshore_regions.loc[regions.index, "cluster"] = country + " 0"
                    continue
                intesections = get_region_intersections(regions)
                w = libpysal.weights.W(intesections)
                model = Spenc(regions, w, n_clusters=n_cluster, attrs_name=["cf"])
                model.solve()
                offshore_regions.loc[regions.index, "cluster"] = (
                    country + " " + model.labels_.astype("str").astype("object")
                )

            cluster_centroids = center_of_mass(
                offshore_regions, groupby="cluster", weight="yield"
            )
            cluster_map = offshore_regions.cluster
        elif "p-h" in offgrid:
            p_nom_max = int(offgrid.split("-")[0]) * 1e3
            intesections = get_region_intersections(offshore_regions.reset_index())
            w = libpysal.weights.W(intesections)

            model = MaxPHeuristic(
                offshore_regions.reset_index(),
                w,
                attrs_name=["cf"],
                threshold_name="p_nom_max",
                threshold=p_nom_max,
            )
            model.solve()
            offshore_regions["hub"] = np.array(model.labels_)
            hub_location = center_of_mass(
                offshore_regions, groupby="hub", weight="yield"
            )
            # connect only regions to hub which are closer to the hub than to onshore node
            coords = coords.loc[offshore_regions.index, :]
            coords["hub"] = list(
                map(tuple, (hub_location.loc[offshore_regions.hub].values))
            )
            offshore_regions["distance_hub"] = coords.apply(
                lambda x: geodesic(x.offshore, x.hub).km, axis=1
            )
            offshore_regions.loc[
                offshore_regions["distance_hub"] > offshore_regions["distance"], "hub"
            ] = None
            # ax = offshore_regions.plot(column='cluster', categorical=True, edgecolor='w', legend=True, cmap="tab20c", figsize=(20,20))
            # ax.scatter(hub_location["x"], hub_location["y"])
        elif "h" in offgrid:
            n_clusters = int(offgrid.split("-")[0])
            intesections = get_region_intersections(offshore_regions.reset_index())
            w = libpysal.weights.W(intesections)
            model = Spenc(
                offshore_regions,
                w,
                n_clusters=n_clusters,
                attrs_name=["cf"],
            )
            model.solve()
            offshore_regions["hub"] = np.array(model.labels_)
            hub_location = center_of_mass(
                offshore_regions, groupby="hub", weight="yield"
            )
            # connect only regions to hub which are closer to the hub than to onshore node
            coords = coords.loc[offshore_regions.index, :]
            coords["hub"] = list(
                map(tuple, (hub_location.loc[offshore_regions.hub].values))
            )
            offshore_regions["distance_hub"] = coords.apply(
                lambda x: geodesic(x.offshore, x.hub).km, axis=1
            )
            offshore_regions.loc[
                offshore_regions["distance_hub"] > offshore_regions["distance"], "hub"
            ] = None

        # Add offshore buses for offshore regions
        if "c" in offgrid:
            n.madd(
                "Bus",
                names="off_" + cluster_centroids.index.values,
                v_nom=220,
                x=cluster_centroids["x"].values,
                y=cluster_centroids["y"].values,
                substation_off=True,
                country=cluster_centroids.index.str[:2].values,
            )
        elif "h" in offgrid:
            n.madd(
                "Bus",
                names="hub_" + hub_location.index.astype("str").values,
                v_nom=220,
                x=hub_location["x"].values,
                y=hub_location["y"].values,
                substation_off=True,
            )
            cluster_map = None
        elif offgrid == "all":
            n.madd(
                "Bus",
                names="off_" + offshore_regions.index,
                v_nom=220,
                x=offshore_regions["x"].values,
                y=offshore_regions["y"].values,
                substation_off=True,
                country=offshore_regions["country"].values,
            )
            cluster_map = None

        # move offshore generators to offshore buses
        move_generators(offshore_regions, cluster_map)

        add_offshore_connections()

        n.export_to_netcdf(snakemake.output[0])
