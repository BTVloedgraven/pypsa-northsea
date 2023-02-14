import logging

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging
from sklearn.neighbors import BallTree
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt

def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and.

    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6
    """

    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n

def load_costs(tech_costs, config, elec_config, Nyears=1.0):
    # set all asset costs and other parameters
    costs = pd.read_csv(tech_costs, index_col=[0, 1]).sort_index()

    # correct units to MW
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW")

    fill_values = config["fill_values"]
    costs = costs.value.unstack().fillna(fill_values)

    costs["capital_cost"] = (
        (
            calculate_annuity(costs["lifetime"], costs["discount rate"])
            + costs["FOM"] / 100.0
        )
        * costs["investment"]
        * Nyears
    )

    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    costs.at["OCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]
    costs.at["CCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]

    costs.at["solar", "capital_cost"] = (
        config["rooftop_share"] * costs.at["solar-rooftop", "capital_cost"]
        + (1 - config["rooftop_share"]) * costs.at["solar-utility", "capital_cost"]
    )

    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(
            dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0)
        )

    max_hours = elec_config["max_hours"]
    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"],
        costs.loc["battery inverter"],
        max_hours=max_hours["battery"],
    )
    costs.loc["H2"] = costs_for_storage(
        costs.loc["hydrogen storage underground"],
        costs.loc["fuel cell"],
        costs.loc["electrolysis"],
        max_hours=max_hours["H2"],
    )

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

    return costs

def add_offshore_connections(
    n,
    costs,
):
    # Create line for every offshore bus and connect it to onshore buses
    onshore_coords = n.buses.loc[~n.buses.index.str.contains("off"), ["x", "y"]]
    offshore_coords = n.buses.loc[n.buses.index.str.contains("off"), ["x", "y"]]

    coords = pd.concat([onshore_coords, offshore_coords])
    coords["xy"] = list(map(tuple, (coords[["x", "y"]]).values))
    on_line_graph = nx.Graph()
    for i, bus0 in enumerate(offshore_coords.index):
        for j, bus1 in enumerate(onshore_coords.index):
            on_line_graph.add_edge(bus0, bus1)

    lines_df = nx.to_pandas_edgelist(on_line_graph)

    lines_df = lines_df.rename(
        columns={"source": "bus0", "target": "bus1"}
    ).astype({"bus0": "string", "bus1": "string"})

    lines_df.loc[:, "length"] = lines_df.apply(
        lambda x: geodesic(coords.loc[x.bus0, "xy"], coords.loc[x.bus1, "xy"]).km,
        axis=1,
    )
    lines_df.drop(lines_df.query("length==0").index, inplace=True)
    lines_df.index = "off_" + lines_df.index.astype("str")

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

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_electricity")
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )

    add_offshore_connections(
        n,
        costs,
    )
    
    n.export_to_netcdf(snakemake.output[0])