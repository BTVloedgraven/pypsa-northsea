# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import logging

import numpy as np
import pandas as pd
import libpysal
import pypsa
import xarray as xr
from _helpers import configure_logging
from add_electricity import (
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    load_costs,
)
from sklearn.neighbors import BallTree
import networkx as nx
from geopy.distance import geodesic

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def attach_storageunits(n, costs, elec_opts):
    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    max_hours = elec_opts["max_hours"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    buses_i = buses_i[~buses_i.str.contains('off')]

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        roundtrip_correction = 0.5 if carrier == "battery" else 1

        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_store[carrier], "efficiency"]
            ** roundtrip_correction,
            efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"]
            ** roundtrip_correction,
            max_hours=max_hours[carrier],
            cyclic_state_of_charge=True,
        )

def attach_converters(n, costs):
    buses_i = n.buses.index
    buses_i = buses_i[buses_i.str.contains('off')]
    buses_i = buses_i[~buses_i.str.contains('H2')]
    bus_sub_dict = {k: n.buses.loc[buses_i][k].values for k in ["x", "y", "country"]}
    HVAC_buses_i = n.madd("Bus", buses_i + " AC", carrier="AC", **bus_sub_dict)
    HVDC_buses_i = n.madd("Bus", buses_i + " DC", carrier="DC", **bus_sub_dict)

    n.madd(
            "Link",
            buses_i + " AC/HVAC Converter",
            bus0=buses_i,
            bus1=HVAC_buses_i,
            carrier="AC/HVAC",
            p_nom_extendable=True,
            capital_cost=0.75*costs.at["offwind-ac-station", "capital_cost"],
        )
    n.madd(
            "Link",
            buses_i + " HVAC/AC Converter",
            bus0=HVAC_buses_i,
            bus1=buses_i,
            carrier="HVAC/AC",
            p_nom_extendable=True,
        )
    n.madd(
            "Link",
            buses_i + " AC/HVDC Converter",
            bus0=buses_i,
            bus1=HVDC_buses_i,
            carrier="AC/HVDC",
            p_nom_extendable=True,
            capital_cost=0.75*costs.at["offwind-dc-station", "capital_cost"],
        )
    n.madd(
            "Link",
            buses_i + " HVDC/AC Converter",
            bus0=HVDC_buses_i,
            bus1=buses_i,
            carrier="HVDC/AC",
            p_nom_extendable=True,
        )
    
    cost_per_m_depth = snakemake.config['costs']['dc_station_depth_cost']
    reference_depth = snakemake.config['costs']['dc_station_reference_depth']
    capex_depth = cost_per_m_depth*(-n.buses["water_depth"]-reference_depth) # in €/MW

    capital_costs_depth = (
        capex_depth
        * (
            calculate_annuity(
                costs.at['offwind-dc-station', "lifetime"],
                costs.at['offwind-dc-station', "discount rate"],
            )
            + costs.at['offwind-dc-station', "FOM"] / 100.0
        )
        * Nyears
    )

    n.links.loc[buses_i + " AC/HVAC Converter", "capital_cost"] = 0.75*costs.at["offwind-ac-station", "capital_cost"] + n.links.loc[buses_i + " AC/HVAC Converter", "bus0"].apply(
        lambda x: capital_costs_depth.loc[x]
    )

    n.links.loc[buses_i + " AC/HVDC Converter", "capital_cost"] = 0.75*costs.at["offwind-dc-station", "capital_cost"] + n.links.loc[buses_i + " AC/HVDC Converter", "bus0"].apply(
        lambda x: capital_costs_depth.loc[x]
    )

def attach_stores(n, costs, elec_opts):
    carriers = elec_opts["extendable_carriers"]["Store"]

    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index
    buses_on_i = buses_i[~buses_i.str.contains('off')]
    buses_off_i = buses_i[buses_i.str.contains('off')]
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)
        h2_buses_on_i = h2_buses_i[~h2_buses_i.str.contains('off')]
        h2_buses_off_i = h2_buses_i[h2_buses_i.str.contains('off')]

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen underground storage", "capital_cost"],
        )

        n.madd(
            "Link",
            h2_buses_on_i + " Electrolysis",
            bus0=buses_on_i,
            bus1=h2_buses_on_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
        )

        offshore_electrolysis = snakemake.config['offshore_options'].get('offshore-electrolysis', True)
        if offshore_electrolysis:
            n.madd(
                "Link",
                h2_buses_off_i + " Electrolysis",
                bus0=buses_off_i,
                bus1=h2_buses_off_i,
                carrier="H2 electrolysis",
                p_nom_extendable=True,
                efficiency=costs.at["electrolysis offshore", "efficiency"],
                capital_cost=costs.at["electrolysis offshore", "capital_cost"],
                marginal_cost=costs.at["electrolysis offshore", "marginal_cost"],
            )

        n.madd(
            "Link",
            h2_buses_on_i + " Fuel Cell",
            bus0=h2_buses_on_i,
            bus1=buses_on_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            # NB: fixed cost is per MWel
            capital_cost=costs.at["fuel cell", "capital_cost"]
            * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
        )

    if "battery" in carriers:
        b_buses_i = n.madd(
            "Bus", buses_on_i + " battery", carrier="battery", **bus_sub_dict
        )

        n.madd(
            "Store",
            b_buses_i,
            bus=b_buses_i,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["battery storage", "capital_cost"],
            marginal_cost=costs.at["battery", "marginal_cost"],
        )

        n.madd(
            "Link",
            b_buses_i + " charger",
            bus0=buses_i,
            bus1=b_buses_i,
            carrier="battery charger",
            # the efficiencies are "round trip efficiencies"
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

        n.madd(
            "Link",
            b_buses_i + " discharger",
            bus0=b_buses_i,
            bus1=buses_i,
            carrier="battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )


def attach_hydrogen_pipelines(n, costs, elec_opts, offshore_grid):
    ext_carriers = elec_opts["extendable_carriers"]
    as_stores = ext_carriers.get("Store", [])

    if "H2 pipeline" not in ext_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines.loc[~n.lines.bus0.str.contains("off") & ~n.lines.bus1.str.contains("off")][attrs], 
         n.links.loc[~n.links.bus0.str.contains("off") & ~n.links.bus1.str.contains("off") & n.links.length != 0][attrs],
         offshore_grid]
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]

    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)
    off_h2_links = h2_links.loc[h2_links.index.str.contains('off')]
    on_h2_links = h2_links.loc[~h2_links.index.str.contains('off')]

    # add pipelines
    n.madd(
        "Link",
        on_h2_links.index,
        bus0=on_h2_links.bus0.values + " H2",
        bus1=on_h2_links.bus1.values + " H2",
        p_nom_extendable=True,
        p_min_pu = -1,
        efficiency = 1,
        length=on_h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * on_h2_links.length,
        carrier="H2 pipeline",
    )
    if snakemake.config['offshore_options'].get('offshore-h2-pipelines', True) and not snakemake.config['offshore_options'].get('only-radial', True):
        n.madd(
            "Link",
            off_h2_links.index,
            bus0=off_h2_links.bus0.values + " H2",
            bus1=off_h2_links.bus1.values + " H2",
            p_nom_extendable=True,
            p_min_pu = -1,
            efficiency = 1,
            length=off_h2_links.length.values,
            capital_cost=costs.at["H2 pipeline offshore", "capital_cost"] * off_h2_links.length,
            carrier="H2 pipeline",
        )

def offshore_grid(n):
    # Create line for every offshore bus and connect it to onshore buses
    onshore_buses = snakemake.config['offshore_options']['onshore-connection-buses']
    onshore_coords = n.buses.loc[n.buses.index.isin(onshore_buses), ["x", "y"]]
    # onshore_coords = n.buses.loc[~n.buses.index.str.contains("off"), ["x", "y"]]
    offshore_coords = n.buses.loc[n.buses.index.str.contains("off") & ~n.buses.index.str.contains("AC|DC|H2"), ["x", "y"]]

    coords = pd.concat([onshore_coords, offshore_coords])

    # works better than with closest neighbors. maybe only create graph like this for offshore buses:
    cells, generators = libpysal.cg.voronoi_frames(
        coords.values, clip="convex hull"
    )
    delaunay = libpysal.weights.Rook.from_dataframe(cells)
    offshore_line_graph = delaunay.to_networkx()
    offshore_line_graph = nx.relabel_nodes(
        offshore_line_graph, dict(zip(offshore_line_graph, coords.index))
    )

    lines_df = nx.to_pandas_edgelist(offshore_line_graph)

    lines_df = lines_df.rename(
        columns={"source": "bus0", "target": "bus1", "weight": "length"}
    ).astype({"bus0": "string", "bus1": "string", "length": "float"})
    lines_df.drop(lines_df.loc[~lines_df.bus0.str.contains('off') & ~lines_df.bus1.str.contains('off')].index, inplace=True)
    coords["latlon"] = list(map(tuple, (coords[["y", "x"]]).values))

    lines_df.loc[:, "length"] = lines_df.apply(
        lambda x: geodesic(coords.loc[x.bus0, "latlon"], coords.loc[x.bus1, "latlon"]).km,
        axis=1,
    )
    lines_df.drop(lines_df.query("length==0").index, inplace=True)

    return lines_df

def add_AC_connections(
    n,
    costs,
    offshore_grid,
):
    lines_df = offshore_grid.copy()
    lines_df.index = "off_AC_" + lines_df.index.astype("str")
    lines_df['bus0'] = lines_df['bus0'].apply(lambda x: x + " AC" if "off" in x else x)
    lines_df['bus1'] = lines_df['bus1'].apply(lambda x: x + " AC" if "off" in x else x)

    n.madd(
        "Line",
        names=lines_df.index,
        bus0=lines_df["bus0"].values,
        bus1=lines_df["bus1"].values,
        length=lines_df["length"].values,
        type = 'Al/St 240/40 4-bundle 380.0',
        num_parallel = 0
    )
    # attach cable cost AC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = n.lines.loc[lines_df.index, "length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["offwind-ac-connection-submarine", "capital_cost"]
    )
    station_cost = n.lines.loc[lines_df.index].apply(lambda x: 0 if (('off' in x.bus0) and ('off' in x.bus1)) else 0.25*costs.at["offwind-ac-station", "capital_cost"], axis=1)
    n.lines.loc[lines_df.index, "capital_cost"] = cable_cost + station_cost

def add_DC_connections(
    n,
    costs,
    offshore_grid,
    all_radial_connections,
):
    links_df = offshore_grid.copy().reset_index(drop=True)
    all_radial_connections = all_radial_connections.copy()
    if snakemake.config['offshore_options'].get('all-dc-radial', True): 
        links_df = pd.concat([links_df, all_radial_connections]).reset_index(drop=True)
    
    links_df['bus0'] = links_df['bus0'].apply(lambda x: x + " DC" if "off" in x else x)
    links_df['bus1'] = links_df['bus1'].apply(lambda x: x + " DC" if "off" in x else x)

    links_df = links_df[
        ~pd.DataFrame(np.sort(links_df[["bus0", "bus1"]])).duplicated()
    ]

    links_df.index = "off_DC_hub_" + links_df.index.astype("str")

    n.madd(
        "Link",
        names=links_df.index,
        carrier="DC",
        p_min_pu = -1,
        efficiency = 1,
        bus0=links_df["bus0"].values,
        bus1=links_df["bus1"].values,
        length=links_df["length"].values,
    )
    # attach cable cost DC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = n.links.loc[links_df.index, "length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["offwind-dc-connection-submarine", "capital_cost"]
    )

    station_cost = n.links.loc[links_df.index].apply(lambda x: 0 if (('off' in x.bus0) and ('off' in x.bus1)) else 0.25*costs.at["offwind-dc-station", "capital_cost"], axis=1)
    n.links.loc[links_df.index, "capital_cost"] = cable_cost + station_cost

def all_radial_connections(
    n,    
):
    # Create line for every offshore bus and connect it to onshore buses
    onshore_buses = snakemake.config['offshore_options']['onshore-connection-buses']
    onshore_coords = n.buses.loc[n.buses.index.isin(onshore_buses), ["x", "y"]]
    offshore_coords = n.buses.loc[n.buses.index.str.contains("off") & ~n.buses.index.str.contains("AC|DC|H2"), ["x", "y"]]

    coords = pd.concat([onshore_coords, offshore_coords])
    coords["latlon"] = list(map(tuple, (coords[["y", "x"]]).values))

    links_df = pd.merge(pd.DataFrame({'bus0': offshore_coords.index}), pd.DataFrame({'bus1': onshore_buses}), how='cross')

    links_df.loc[:, "length"] = links_df.apply(
        lambda x: geodesic(coords.loc[x.bus0, "latlon"], coords.loc[x.bus1, "latlon"]).km,
        axis=1,
    )
    links_df.drop(links_df.query("length==0").index, inplace=True)

    return links_df

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

def old_add_DC_connections(
    n,
    costs,
):
    # Create line for every offshore bus and connect it to onshore buses
    onshore_buses = snakemake.config['offshore_options']['onshore-connection-buses']
    onshore_coords = n.buses.loc[n.buses.index.isin(onshore_buses), ["x", "y"]]
    offshore_coords = n.buses.loc[n.buses.index.str.contains("off" and "DC"), ["x", "y"]]

    coords = pd.concat([onshore_coords, offshore_coords])
    coords["latlon"] = list(map(tuple, (coords[["y", "x"]]).values))

    links_df = pd.merge(pd.DataFrame({'bus0': offshore_coords.index}), pd.DataFrame({'bus1': onshore_buses}), how='cross')
    # links_df_to_shore = pd.merge(pd.DataFrame({'bus0': offshore_coords.index}), pd.DataFrame({'bus1': onshore_buses}), how='cross')
    # import itertools
    # links_df = pd.DataFrame(itertools.combinations(offshore_coords.index, 2), columns=['bus0', 'bus1'])
    # links_df = pd.concat([links_df_to_shore, links_df], ignore_index=True)
    # same_bus = links_df.bus0 == links_df.bus1
    # links_df.drop(links_df.loc[same_bus].index)
    links_df.loc[:, "length"] = links_df.apply(
        lambda x: geodesic(coords.loc[x.bus0, "latlon"], coords.loc[x.bus1, "latlon"]).km,
        axis=1,
    )
    links_df.drop(links_df.query("length==0").index, inplace=True)
    links_df.index = "off_DC_" + links_df.index.astype("str")

    n.madd(
        "Link",
        names=links_df.index,
        carrier="DC",
        p_min_pu = -1,
        efficiency = 1,
        bus0=links_df["bus0"].values,
        bus1=links_df["bus1"].values,
        length=links_df["length"].values,
    )
    # attach cable cost DC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = n.links.loc[links_df.index, "length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["offwind-dc-connection-submarine", "capital_cost"]
    )

    station_cost = n.links.loc[links_df.index].apply(lambda x: 0 if (('off' in x.bus0) and ('off' in x.bus1)) else 0.25*costs.at["offwind-dc-station", "capital_cost"], axis=1)
    n.links.loc[links_df.index, "capital_cost"] = cable_cost + station_cost

def add_radial_connections(
    n,
    costs,
):
    # Create line for every offshore bus and connect it to onshore buses
    onshore_buses = snakemake.config['offshore_options']['onshore-connection-buses']
    onshore_coords = n.buses.loc[n.buses.index.isin(onshore_buses), ["x", "y"]]
    offshore_coords = n.buses.loc[n.buses.index.str.contains("off" and "DC"), ["x", "y"]]
    offshore_coords.index = offshore_coords.index.str.replace(" DC", "")

    # coords = pd.concat([onshore_coords, offshore_coords])
    offshore_coords["latlon"] = list(map(tuple, (offshore_coords[["y", "x"]]).values))
    onshore_coords["latlon"] = list(map(tuple, (onshore_coords[["y", "x"]]).values))

    

    def get_closest(offshore_bus):
        # offshore_bus = offshore_bus.name
        options_df = onshore_coords
        options_df.loc[:, "length"] = options_df.apply(
            lambda x: geodesic(offshore_coords.loc[offshore_bus, "latlon"], x.latlon).km,
            axis=1,
        )
        onshore_bus = options_df["length"].idxmin()
        new_connection = pd.Series({"bus0": offshore_bus, "bus1": onshore_bus, "length": options_df.loc[onshore_bus, "length"]})
        return new_connection

    links_df = offshore_coords.index.to_series().apply(get_closest)

    links_df.drop(links_df.query("length==0").index, inplace=True)
    links_df.index = "radial_" + links_df.index.astype("str")
    links_df["bus0DC"] = links_df.bus0.apply(lambda x: x + " DC")
    links_df["bus0AC"] = links_df.bus0.apply(lambda x: x + " AC")

    n.madd(
        "Link",
        names=links_df.index,
        carrier="DC",
        p_min_pu = -1,
        efficiency = 1,
        bus0=links_df["bus0DC"].values,
        bus1=links_df["bus1"].values,
        length=links_df["length"].values,
    )
    # attach cable cost DC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = n.links.loc[links_df.index, "length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["offwind-dc-connection-submarine", "capital_cost"]
    )

    station_cost = n.links.loc[links_df.index].apply(lambda x: 0 if (('off' in x.bus0) and ('off' in x.bus1)) else 0.25*costs.at["offwind-dc-station", "capital_cost"], axis=1)
    n.links.loc[links_df.index, "capital_cost"] = cable_cost + station_cost

    n.madd(
        "Line",
        names=links_df.index,
        bus0=links_df["bus0AC"].values,
        bus1=links_df["bus1"].values,
        length=links_df["length"].values,
        type = 'Al/St 240/40 4-bundle 380.0',
        num_parallel = 0
    )
    # attach cable cost AC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = n.lines.loc[links_df.index, "length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["offwind-ac-connection-submarine", "capital_cost"]
    )
    station_cost = n.lines.loc[links_df.index].apply(lambda x: 0 if (('off' in x.bus0) and ('off' in x.bus1)) else 0.25*costs.at["offwind-ac-station", "capital_cost"], axis=1)
    n.lines.loc[links_df.index, "capital_cost"] = cable_cost + station_cost

def attach_gas_turbines(n, costs):
    buses_i = n.buses.loc[~n.buses.index.str.contains("off|H2")].index
    CCGT_buses = n.generators.loc[n.generators.carrier == "CCGT"].bus
    OCGT_buses = n.generators.loc[n.generators.carrier == "OCGT"].bus
    for i in buses_i:
        if i not in CCGT_buses.values:
            n.add("Generator",
                i + " CCGT",
                bus = i,
                carrier="CCGT",
                p_nom_extendable=True,
                efficiency=costs.at["CCGT", "efficiency"],
                capital_cost=costs.at["CCGT", "capital_cost"],
                marginal_cost=costs.at["CCGT", "marginal_cost"],
            )
        if i not in OCGT_buses.values:
            n.add("Generator",
                i + " OCGT",
                bus = i,
                carrier="OCGT",
                p_nom_extendable=True,
                efficiency=costs.at["OCGT", "efficiency"],
                capital_cost=costs.at["OCGT", "capital_cost"],
                marginal_cost=costs.at["OCGT", "marginal_cost"],
            )
    return

def attach_GT_to_hydrogen(n, legacy_gt_to_hydrogen):
    
    carriers = legacy_gt_to_hydrogen
    
    gas_turbines = n.generators.query("carrier in @carriers")
    n.mremove("Generator", gas_turbines.index)

    buses_i = gas_turbines.bus
    h2_buses_i = gas_turbines.bus + " H2"

    n.madd("Link",
        gas_turbines.index,
        bus0=h2_buses_i,
        bus1=buses_i,
        carrier=gas_turbines.carrier,
        p_nom_extendable=gas_turbines.p_nom_extendable,
        p_nom=gas_turbines.p_nom,
        p_nom_max=gas_turbines.p_nom_max,
        efficiency=gas_turbines.efficiency,
        #NB: fixed cost is per MWel
        capital_cost=gas_turbines.capital_cost * gas_turbines.efficiency,
        marginal_cost=gas_turbines.marginal_cost * gas_turbines.efficiency,
    )

def attach_hydrogen_loads(n, enable_config):
    
    h2fp = enable_config['hydrogen_contant_loads_at_nodes']
    try:
        h2df = pd.read_csv(h2fp, index_col=0)   
    except:
        logger.warning(f"Invalid hydrogen load filepath supplied: '{h2fp}'")
        return None
    
    if h2df.empty:
        logging.info(f"Not adding hydrogen demands since the demands supplied are empty")
        return None
    else:
        bus_string = ""
        for col in h2df.index:
            bus = col + " H2"
            if not (bus in n.stores.index):
                raise ValueError(f"Hydrogen bus does not exist in network: {bus}.")

            else:
                value = h2df.at[col, h2df.columns[0]]
                bus_string += f"\t{bus}, {int(value)} MW\n"

        h2df.index = [str(i) + " H2" for i in h2df.index]
        logging.info(f"Adding constant hydrogen demands:\n{bus_string}")
        n.madd(
            "Load",
            suffix=" load",
            names=h2df.index,
            carrier="H2",
            bus=h2df.index,
            p_set=h2df.constant_power_MW,
        ) 

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_extra_components", simpl="", clusters=5)
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config["electricity"]
    legacy_gt_to_hydrogen = snakemake.config['enable'].get("legacy_gt_to_hydrogen", False)

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs, snakemake.config["costs"], elec_config, Nyears
    )

    attach_storageunits(n, costs, elec_config)
    attach_stores(n, costs, elec_config)
    attach_converters(n, costs)
    attach_gas_turbines(n, costs)
    attach_GT_to_hydrogen(n, legacy_gt_to_hydrogen)
    attach_hydrogen_loads(n, snakemake.config['enable'])

    offshore_grid = offshore_grid(n)
    all_radial_connections = all_radial_connections(n)

    if snakemake.config['offshore_options'].get('only-radial', True):    
        add_radial_connections(
            n,
            costs,
        )
    else:
        if snakemake.config['offshore_options'].get('ac-grid', True):    
            add_AC_connections(
                n,
                costs,
                offshore_grid,
            )

        if snakemake.config['offshore_options'].get('dc-grid', True): 
            add_DC_connections(
                n,
                costs,
                offshore_grid,
                all_radial_connections,
            )

    attach_hydrogen_pipelines(n, costs, elec_config, offshore_grid)

    add_nice_carrier_names(n, snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
