# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Adds electrical generators and existing hydro storage units to a base network.

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
        conventional_carriers:
        co2limit:
        extendable_carriers:
        estimate_renewable_capacities:


    load:
        scaling_factor:

    renewable:
        hydro:
            carriers:
            hydro_max_hours:
            hydro_capital_cost:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/bundle/hydro_capacities.csv``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.

    .. image:: ../img/hydrocapacities.png
        :scale: 34 %

- ``data/geth2015_hydro_capacities.csv``: alternative to capacities above; not currently used!
- ``resources/load.csv`` Hourly per-country load profiles.
- ``resources/regions_onshore.geojson``: confer :ref:`busregions`
- ``resources/nuts3_shapes.geojson``: confer :ref:`shapes`
- ``resources/powerplants.csv``: confer :ref:`powerplants`
- ``resources/profile_{}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`.
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``networks/elec.nc``:

    .. image:: ../img/elec.png
            :scale: 33 %

Description
-----------

The rule :mod:`add_electricity` ties all the different data inputs from the preceding rules together into a detailed PyPSA network that is stored in ``networks/elec.nc``. It includes:

- today's transmission topology and transfer capacities (optionally including lines which are under construction according to the config settings ``lines: under_construction`` and ``links: under_construction``),
- today's thermal and hydro power generation capacities (for the technologies listed in the config setting ``electricity: conventional_carriers``), and
- today's load time-series (upsampled in a top-down approach according to population and gross domestic product)

It further adds extendable ``generators`` with **zero** capacity for

- photovoltaic, onshore and AC- as well as DC-connected offshore wind installations with today's locational, hourly wind and solar capacity factors (but **no** current capacities),
- additional open- and combined-cycle gas turbines (if ``OCGT`` and/or ``CCGT`` is listed in the config setting ``electricity: extendable_carriers``)
"""

import logging

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import xarray as xr
from _helpers import configure_logging, update_p_nom_max
from powerplantmatching.export import map_country_bus
from vresutils import transfer as vtransfer
from sklearn.neighbors import BallTree
import networkx as nx
from geopy.distance import geodesic

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def normed(s):
    return s / s.sum()


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


def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty:
        return

    emissions_cols = (
        costs.columns.to_series().loc[lambda s: s.str.endswith("_emissions")].values
    )
    suptechs = missing_carriers.str.split("-").str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.0)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, "Carrier")


def calculate_offwind_cost(WD, MW=12, D=236, HH=138, SP=343, DT=8):
    """
    Calculating offshore wind capex considering the average water depth of the
    region. Equations and default values from DEA technology data (https://ens.
    dk/sites/ens.dk/files/Statistik/technology_catalogue_offshore_wind_march_20
    22_-_annex_to_prediction_of_performance_and_cost.pdf).

    Parameters
    ----------
    WD : xarray
        Average water depth of the different regions
    MW : float
        Power of the wind turbine in MW
    D: int
        Rotor diameter of wind turbine in meters
    HH: int
        Hub height of wind turbine in meters
    SP: int
        Specific power of wind turbine in W/m2
    DT: int
        Distance between the wind turbine in number of rotor diameters

    Returns
    -------
    capex: xarray
        Capex of the wind turbine in the different regions
    """
    RA = (D / 2) ** 2 * np.pi
    IA = DT * D
    wind_turbine_invest = (
        -0.6 * SP + 750 + (0.53 * HH * RA + 5500) / (1000 * MW)
    ) * 1.1
    wind_turbine_install = 300 * MW ** (-0.6)
    foundation_invest = (8 * np.abs(WD) + 30) * (
        1 + (0.003 * (350 - np.min([400, SP])))
    )
    foundation_install = 2.5 * np.abs(WD) + 600 * MW ** (-0.6)
    array_cable = IA * 500 / MW / 1000
    turbine_transport = 50
    insurance = 100
    finance_cost = 100
    continences = 50
    development_cost = 0.02  # in % of capex
    capex = (
        np.sum(
            [
                wind_turbine_invest,
                wind_turbine_install,
                foundation_invest,
                foundation_install,
                array_cable,
                turbine_transport,
                insurance,
                finance_cost,
                continences,
            ]
        )
        * (1 + development_cost)
        * 1000
    )  # in €/MW
    return capex


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


def load_powerplants(ppl_fn):
    carrier_dict = {
        "ocgt": "OCGT",
        "ccgt": "CCGT",
        "bioenergy": "biomass",
        "ccgt, thermal": "CCGT",
        "hard coal": "coal",
    }
    return (
        pd.read_csv(ppl_fn, index_col=0, dtype={"bus": "str"})
        .powerplant.to_pypsa_names()
        .rename(columns=str.lower)
        .replace({"carrier": carrier_dict})
    )


def attach_load(n, regions, load, nuts3_shapes, countries, scaling=1.0):
    substation_lv_i = n.buses.index[n.buses["substation_lv"]]
    regions = gpd.read_file(regions).set_index("name").reindex(substation_lv_i)
    opsd_load = pd.read_csv(load, index_col=0, parse_dates=True).filter(items=countries)

    logger.info(f"Load data scaled with scalling factor {scaling}.")
    opsd_load *= scaling

    nuts3 = gpd.read_file(nuts3_shapes).set_index("index")

    def upsample(cntry, group):
        l = opsd_load[cntry]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            nuts3_cntry = nuts3.loc[nuts3.country == cntry]
            transfer = vtransfer.Shapes2Shapes(
                group, nuts3_cntry.geometry, normed=False
            ).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(nuts3_cntry["gdp"].fillna(1.0).values), index=group.index
            )
            pop_n = pd.Series(
                transfer.dot(nuts3_cntry["pop"].fillna(1.0).values), index=group.index
            )

            # relative factors 0.6 and 0.4 have been determined from a linear
            # regression on the country to continent load data
            factors = normed(0.6 * normed(gdp_n) + 0.4 * normed(pop_n))
            return pd.DataFrame(
                factors.values * l.values[:, np.newaxis],
                index=l.index,
                columns=factors.index,
            )

    load = pd.concat(
        [
            upsample(cntry, group)
            for cntry, group in regions.geometry.groupby(regions.country)
        ],
        axis=1,
    )

    n.madd("Load", substation_lv_i, bus=substation_lv_i, p_set=load)


def update_transmission_costs(n, costs, length_factor=1.0):
    # TODO: line length factor of lines is applied to lines and links.
    # Separate the function to distinguish.

    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    if n.links.empty:
        return

    dc_b = n.links.carrier == "DC"

    # If there are no dc links, then the 'underwater_fraction' column
    # may be missing. Therefore we have to return here.
    if n.links.loc[dc_b].empty:
        return

    costs = (
        n.links.loc[dc_b, "length"]
        * length_factor
        * (
            (1.0 - n.links.loc[dc_b, "underwater_fraction"])
            * costs.at["HVDC overhead", "capital_cost"]
            + n.links.loc[dc_b, "underwater_fraction"]
            * costs.at["HVDC submarine", "capital_cost"]
        )
        + costs.at["HVDC inverter pair", "capital_cost"]
    )
    n.links.loc[dc_b, "capital_cost"] = costs


def attach_wind_and_solar(
    n,
    costs,
    input_profiles,
    technologies,
    extendable_carriers,
    config,
    line_length_factor=1,
):
    # TODO: rename tech -> carrier, technologies -> carriers
    _add_missing_carriers_from_costs(n, costs, technologies)

    for tech in technologies:
        if tech == "hydro":
            continue

        with xr.open_dataset(getattr(input_profiles, "profile_" + tech)) as ds:
            if ds.indexes["bus"].empty:
                continue

            suptech = tech.split("-", 2)[0]
            if suptech == "offwind":
                underwater_fraction = ds["underwater_fraction"].to_pandas()
                cable_cost = (
                    line_length_factor
                    * ds["average_distance"].to_pandas()
                    * (
                        underwater_fraction
                        * costs.at[tech + "-connection-submarine", "capital_cost"]
                        + (1.0 - underwater_fraction)
                        * costs.at[tech + "-connection-underground", "capital_cost"]
                    )
                )
                grid_connection_cost = (
                    costs.at[tech + "-station", "capital_cost"] + cable_cost
                )
                calculate_topology_cost = config[tech].get(
                    "calculate_topology_cost", False
                )
                if calculate_topology_cost and tech != "offwind-float":
                    import atlite

                    turbine_type = config[tech]["resource"]["turbine"]
                    turbine_config = atlite.resource.get_windturbineconfig(turbine_type)
                    kwargs = {
                        "WD": ds["water_depth"].to_pandas(),
                        "MW": turbine_config["P"],
                        "HH": turbine_config["hub_height"],
                    }
                    turbine_cost = (
                        calculate_offwind_cost(**kwargs)
                        * (
                            calculate_annuity(
                                costs.at[suptech, "lifetime"],
                                costs.at[suptech, "discount rate"],
                            )
                            + costs.at[suptech, "FOM"] / 100.0
                        )
                        * Nyears
                    )
                else:
                    turbine_cost = costs.at[tech, "capital_cost"]
                capital_cost = turbine_cost + grid_connection_cost

                logger.info(
                    "Added connection cost of {:0.0f}-{:0.0f} Eur/MW/a to {}".format(
                        cable_cost.min(), cable_cost.max(), tech
                    )
                )
                n.madd(
                    "Generator",
                    ds.indexes["bus"],
                    " " + tech,
                    bus=ds.indexes["bus"],
                    carrier=tech,
                    p_nom_extendable=tech in extendable_carriers["Generator"],
                    p_nom_max=ds["p_nom_max"].to_pandas(),
                    weight=ds["weight"].to_pandas(),
                    marginal_cost=costs.at[suptech, "marginal_cost"],
                    capital_cost=capital_cost,
                    grid_connection_cost=grid_connection_cost,
                    turbine_cost=turbine_cost,
                    efficiency=costs.at[suptech, "efficiency"],
                    p_max_pu=ds["profile"].transpose("time", "bus").to_pandas(),
                )
            else:
                capital_cost = costs.at[tech, "capital_cost"]
                n.madd(
                    "Generator",
                    ds.indexes["bus"],
                    " " + tech,
                    bus=ds.indexes["bus"],
                    carrier=tech,
                    p_nom_extendable=True,
                    p_nom_max=ds["p_nom_max"].to_pandas(),
                    weight=ds["weight"].to_pandas(),
                    marginal_cost=costs.at[tech, "marginal_cost"],
                    capital_cost=capital_cost,
                    efficiency=costs.at[tech, "efficiency"],
                    p_max_pu=ds["profile"].transpose("time", "bus").to_pandas(),
                )

def move_generators(
    n,
    costs,
    offshore_regions,
    cluster_map,
):
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

def add_offshore_connections(
    n,
    costs,
):
    # Create line for every offshore bus and connect it to onshore buses
    onshore_coords = n.buses.loc[~n.buses.index.str.contains("off"), ["x", "y"]]
    offshore_coords = n.buses.loc[n.buses.index.str.contains("off"), ["x", "y"]]

    tree = BallTree(
        np.radians(offshore_coords), leaf_size=40, metric="haversine"
    )

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
    # lines_filter = offshore_lines.apply(
    #     lambda x: LineString(
    #         [n.buses.loc[x.source, ["x", "y"]], n.buses.loc[x.target, ["x", "y"]]]
    #     ),
    #     axis=1,
    # )
    # onshore_shape = onshore_regions.unary_union
    # lines_filter = lines_filter.apply(lambda x: x.intersects(onshore_shape))
    # offshore_lines.drop(offshore_lines[lines_filter].index, inplace=True)

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

    n.madd(
        "Line",
        names=lines_df.index,
        v_nom=220,
        bus0=lines_df["bus0"].values,
        bus1=lines_df["bus1"].values,
        length=lines_df["length"].values,
        #type="149-AL1/24-ST1A 110.0",
    )
    # attach cable cost AC for offshore grid lines
    line_length_factor = snakemake.config["lines"]["length_factor"]
    cable_cost = n.lines.loc[lines_df.index, "length"].apply(
        lambda x: x
        * line_length_factor
        * costs.at["offwind-ac-connection-submarine", "capital_cost"]
    )
    n.lines.loc[lines_df.index, "capital_cost"] = cable_cost

    # n.madd(
    #     "Link",
    #     names=lines_df.index,
    #     carrier="DC",
    #     bus0=lines_df["bus0"].values,
    #     bus1=lines_df["bus1"].values,
    #     length=lines_df["length"].values,
    # )
    # # attach cable cost DC for offshore grid lines
    # line_length_factor = snakemake.config["lines"]["length_factor"]
    # cable_cost = n.links.loc[lines_df.index, "length"].apply(
    #     lambda x: x
    #     * line_length_factor
    #     * costs.at["offwind-dc-connection-submarine", "capital_cost"]
    # )
    # n.links.loc[lines_df.index, "capital_cost"] = cable_cost

def attach_conventional_generators(
    n,
    costs,
    ppl,
    conventional_carriers,
    extendable_carriers,
    conventional_config,
    conventional_inputs,
):
    carriers = set(conventional_carriers) | set(extendable_carriers["Generator"])
    _add_missing_carriers_from_costs(n, costs, carriers)

    ppl = (
        ppl.query("carrier in @carriers")
        .join(costs, on="carrier", rsuffix="_r")
        .rename(index=lambda s: "C" + str(s))
    )
    ppl["efficiency"] = ppl.efficiency.fillna(ppl.efficiency_r)
    ppl["marginal_cost"] = (
        ppl.carrier.map(costs.VOM) + ppl.carrier.map(costs.fuel) / ppl.efficiency
    )

    logger.info(
        "Adding {} generators with capacities [GW] \n{}".format(
            len(ppl), ppl.groupby("carrier").p_nom.sum().div(1e3).round(2)
        )
    )

    n.madd(
        "Generator",
        ppl.index,
        carrier=ppl.carrier,
        bus=ppl.bus,
        p_nom_min=ppl.p_nom.where(ppl.carrier.isin(conventional_carriers), 0),
        p_nom=ppl.p_nom.where(ppl.carrier.isin(conventional_carriers), 0),
        p_nom_extendable=ppl.carrier.isin(extendable_carriers["Generator"]),
        efficiency=ppl.efficiency,
        marginal_cost=ppl.marginal_cost,
        capital_cost=ppl.capital_cost,
        build_year=ppl.datein.fillna(0).astype(int),
        lifetime=(ppl.dateout - ppl.datein).fillna(np.inf),
    )

    for carrier in conventional_config:

        # Generators with technology affected
        idx = n.generators.query("carrier == @carrier").index

        for attr in list(set(conventional_config[carrier]) & set(n.generators)):

            values = conventional_config[carrier][attr]

            if f"conventional_{carrier}_{attr}" in conventional_inputs:
                # Values affecting generators of technology k country-specific
                # First map generator buses to countries; then map countries to p_max_pu
                values = pd.read_csv(values, index_col=0).iloc[:, 0]
                bus_values = n.buses.country.map(values)
                n.generators[attr].update(
                    n.generators.loc[idx].bus.map(bus_values).dropna()
                )
            else:
                # Single value affecting all generators of technology k indiscriminantely of country
                n.generators.loc[idx, attr] = values


def attach_hydro(n, costs, ppl, profile_hydro, hydro_capacities, carriers, **config):
    _add_missing_carriers_from_costs(n, costs, carriers)

    ppl = (
        ppl.query('carrier == "hydro"')
        .reset_index(drop=True)
        .rename(index=lambda s: str(s) + " hydro")
    )
    ror = ppl.query('technology == "Run-Of-River"')
    phs = ppl.query('technology == "Pumped Storage"')
    hydro = ppl.query('technology == "Reservoir"')

    country = ppl["bus"].map(n.buses.country).rename("country")

    inflow_idx = ror.index.union(hydro.index)
    if not inflow_idx.empty:
        dist_key = ppl.loc[inflow_idx, "p_nom"].groupby(country).transform(normed)

        with xr.open_dataarray(profile_hydro) as inflow:
            inflow_countries = pd.Index(country[inflow_idx])
            missing_c = inflow_countries.unique().difference(
                inflow.indexes["countries"]
            )
            assert missing_c.empty, (
                f"'{profile_hydro}' is missing "
                f"inflow time-series for at least one country: {', '.join(missing_c)}"
            )

            inflow_t = (
                inflow.sel(countries=inflow_countries)
                .rename({"countries": "name"})
                .assign_coords(name=inflow_idx)
                .transpose("time", "name")
                .to_pandas()
                .multiply(dist_key, axis=1)
            )

    if "ror" in carriers and not ror.empty:
        n.madd(
            "Generator",
            ror.index,
            carrier="ror",
            bus=ror["bus"],
            p_nom=ror["p_nom"],
            efficiency=costs.at["ror", "efficiency"],
            capital_cost=costs.at["ror", "capital_cost"],
            weight=ror["p_nom"],
            p_max_pu=(
                inflow_t[ror.index]
                .divide(ror["p_nom"], axis=1)
                .where(lambda df: df <= 1.0, other=1.0)
            ),
        )

    if "PHS" in carriers and not phs.empty:
        # fill missing max hours to config value and
        # assume no natural inflow due to lack of data
        max_hours = config.get("PHS_max_hours", 6)
        phs = phs.replace({"max_hours": {0: max_hours}})
        n.madd(
            "StorageUnit",
            phs.index,
            carrier="PHS",
            bus=phs["bus"],
            p_nom=phs["p_nom"],
            capital_cost=costs.at["PHS", "capital_cost"],
            max_hours=phs["max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
        )

    if "hydro" in carriers and not hydro.empty:
        hydro_max_hours = config.get("hydro_max_hours")

        assert hydro_max_hours is not None, "No path for hydro capacities given."

        hydro_stats = pd.read_csv(
            hydro_capacities, comment="#", na_values="-", index_col=0
        )
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval("p_nom * max_hours").groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query("max_hours == 0").index

        if hydro_max_hours == "energy_capacity_totals_by_country":
            # watch out some p_nom values like IE's are totally underrepresented
            max_hours_country = (
                e_missing / hydro.loc[missing_mh_i].groupby("country").p_nom.sum()
            )

        elif hydro_max_hours == "estimate_by_large_installations":
            max_hours_country = (
                hydro_stats["E_store[TWh]"] * 1e3 / hydro_stats["p_nom_discharge[GW]"]
            )

        missing_countries = pd.Index(hydro["country"].unique()).difference(
            max_hours_country.dropna().index
        )
        if not missing_countries.empty:
            logger.warning(
                "Assuming max_hours=6 for hydro reservoirs in the countries: {}".format(
                    ", ".join(missing_countries)
                )
            )
        hydro_max_hours = hydro.max_hours.where(
            hydro.max_hours > 0, hydro.country.map(max_hours_country)
        ).fillna(6)

        n.madd(
            "StorageUnit",
            hydro.index,
            carrier="hydro",
            bus=hydro["bus"],
            p_nom=hydro["p_nom"],
            max_hours=hydro_max_hours,
            capital_cost=costs.at["hydro", "capital_cost"],
            marginal_cost=costs.at["hydro", "marginal_cost"],
            p_max_pu=1.0,  # dispatch
            p_min_pu=0.0,  # store
            efficiency_dispatch=costs.at["hydro", "efficiency"],
            efficiency_store=0.0,
            cyclic_state_of_charge=True,
            inflow=inflow_t.loc[:, hydro.index],
        )


def attach_extendable_generators(n, costs, ppl, carriers):
    logger.warning(
        "The function `attach_extendable_generators` is deprecated in v0.5.0."
    )
    _add_missing_carriers_from_costs(n, costs, carriers)

    for tech in carriers:
        if tech.startswith("OCGT"):
            ocgt = (
                ppl.query("carrier in ['OCGT', 'CCGT']")
                .groupby("bus", as_index=False)
                .first()
            )
            n.madd(
                "Generator",
                ocgt.index,
                suffix=" OCGT",
                bus=ocgt["bus"],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.0,
                capital_cost=costs.at["OCGT", "capital_cost"],
                marginal_cost=costs.at["OCGT", "marginal_cost"],
                efficiency=costs.at["OCGT", "efficiency"],
            )

        elif tech.startswith("CCGT"):
            ccgt = (
                ppl.query("carrier in ['OCGT', 'CCGT']")
                .groupby("bus", as_index=False)
                .first()
            )
            n.madd(
                "Generator",
                ccgt.index,
                suffix=" CCGT",
                bus=ccgt["bus"],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.0,
                capital_cost=costs.at["CCGT", "capital_cost"],
                marginal_cost=costs.at["CCGT", "marginal_cost"],
                efficiency=costs.at["CCGT", "efficiency"],
            )

        elif tech.startswith("nuclear"):
            nuclear = (
                ppl.query("carrier == 'nuclear'").groupby("bus", as_index=False).first()
            )
            n.madd(
                "Generator",
                nuclear.index,
                suffix=" nuclear",
                bus=nuclear["bus"],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.0,
                capital_cost=costs.at["nuclear", "capital_cost"],
                marginal_cost=costs.at["nuclear", "marginal_cost"],
                efficiency=costs.at["nuclear", "efficiency"],
            )

        else:
            raise NotImplementedError(
                f"Adding extendable generators for carrier "
                "'{tech}' is not implemented, yet. "
                "Only OCGT, CCGT and nuclear are allowed at the moment."
            )


def attach_OPSD_renewables(n, tech_map):
    tech_string = ", ".join(sum(tech_map.values(), []))
    logger.info(f"Using OPSD renewable capacities for carriers {tech_string}.")

    df = pm.data.OPSD_VRE().powerplant.convert_country_to_alpha2()
    technology_b = ~df.Technology.isin(["Onshore", "Offshore"])
    df["Fueltype"] = df.Fueltype.where(technology_b, df.Technology).replace(
        {"Solar": "PV"}
    )
    df = df.query("Fueltype in @tech_map").powerplant.convert_country_to_alpha2()

    for fueltype, carriers in tech_map.items():
        gens = n.generators[lambda df: df.carrier.isin(carriers)]
        buses = n.buses.loc[gens.bus.unique()]
        gens_per_bus = gens.groupby("bus").p_nom.count()

        caps = map_country_bus(df.query("Fueltype == @fueltype"), buses)
        caps = caps.groupby(["bus"]).Capacity.sum()
        caps = caps / gens_per_bus.reindex(caps.index, fill_value=1)

        n.generators.p_nom.update(gens.bus.map(caps).dropna())
        n.generators.p_nom_min.update(gens.bus.map(caps).dropna())


def estimate_renewable_capacities(n, config):
    year = config["electricity"]["estimate_renewable_capacities"]["year"]
    tech_map = config["electricity"]["estimate_renewable_capacities"][
        "technology_mapping"
    ]
    countries = config["countries"]
    expansion_limit = config["electricity"]["estimate_renewable_capacities"][
        "expansion_limit"
    ]

    if not len(countries) or not len(tech_map):
        return

    capacities = pm.data.IRENASTAT().powerplant.convert_country_to_alpha2()
    capacities = capacities.query(
        "Year == @year and Technology in @tech_map and Country in @countries"
    )
    capacities = capacities.groupby(["Technology", "Country"]).Capacity.sum()

    logger.info(
        f"Heuristics applied to distribute renewable capacities [GW]: "
        f"\n{capacities.groupby('Technology').sum().div(1e3).round(2)}"
    )

    for ppm_technology, techs in tech_map.items():
        tech_i = n.generators.query("carrier in @techs").index
        stats = capacities.loc[ppm_technology].reindex(countries, fill_value=0.0)
        country = n.generators.bus[tech_i].map(n.buses.country)
        existent = n.generators.p_nom[tech_i].groupby(country).sum()
        missing = stats - existent
        dist = n.generators_t.p_max_pu.mean() * n.generators.p_nom_max

        n.generators.loc[tech_i, "p_nom"] += (
            dist[tech_i]
            .groupby(country)
            .transform(lambda s: normed(s) * missing[s.name])
            .where(lambda s: s > 0.1, 0.0)  # only capacities above 100kW
        )
        n.generators.loc[tech_i, "p_nom_min"] = n.generators.loc[tech_i, "p_nom"]

        if expansion_limit:
            assert np.isscalar(expansion_limit)
            logger.info(
                f"Reducing capacity expansion limit to {expansion_limit*100:.2f}% of installed capacity."
            )
            n.generators.loc[tech_i, "p_nom_max"] = (
                expansion_limit * n.generators.loc[tech_i, "p_nom_min"]
            )


def add_nice_carrier_names(n, config):
    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(f"tech_colors for carriers {missing_i} not defined in config.")
    n.carriers["color"] = colors


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_electricity")
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        snakemake.config["electricity"],
        Nyears,
    )
    ppl = load_powerplants(snakemake.input.powerplants)

    if "renewable_carriers" in snakemake.config["electricity"]:
        renewable_carriers = set(snakemake.config["renewable"])
    else:
        logger.warning(
            "Missing key `renewable_carriers` under config entry `electricity`. "
            "In future versions, this will raise an error. "
            "Falling back to carriers listed under `renewable`."
        )
        renewable_carriers = snakemake.config["renewable"]

    extendable_carriers = snakemake.config["electricity"]["extendable_carriers"]
    if not (set(renewable_carriers) & set(extendable_carriers["Generator"])):
        logger.warning(
            "No renewables found in config entry `extendable_carriers`. "
            "In future versions, these have to be explicitly listed. "
            "Falling back to all renewables."
        )

    conventional_carriers = snakemake.config["electricity"]["conventional_carriers"]

    attach_load(
        n,
        snakemake.input.onshore_regions,
        snakemake.input.load,
        snakemake.input.nuts3_shapes,
        snakemake.config["countries"],
        snakemake.config["load"]["scaling_factor"],
    )

    update_transmission_costs(n, costs, snakemake.config["lines"]["length_factor"])

    conventional_inputs = {
        k: v for k, v in snakemake.input.items() if k.startswith("conventional_")
    }
    attach_conventional_generators(
        n,
        costs,
        ppl,
        conventional_carriers,
        extendable_carriers,
        snakemake.config.get("conventional", {}),
        conventional_inputs,
    )

    attach_wind_and_solar(
        n,
        costs,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        snakemake.config["renewable"],
        snakemake.config["lines"]["length_factor"],
    )

    move_generators(
        n,
        costs,
        snakemake.input.offshore_regions, 
        snakemake.input.busmap_offshore,
    )

    add_offshore_connections(
        n,
        costs,
    )

    if "estimate_renewable_capacities" not in snakemake.config["electricity"]:
        logger.warning(
            "Missing key `estimate_renewable_capacities` under config entry `electricity`. "
            "In future versions, this will raise an error. "
            "Falling back to whether ``estimate_renewable_capacities_from_capacity_stats`` is in the config."
        )
        if (
            "estimate_renewable_capacities_from_capacity_stats"
            in snakemake.config["electricity"]
        ):
            estimate_renewable_caps = {
                "enable": True,
                **snakemake.config["electricity"][
                    "estimate_renewable_capacities_from_capacity_stats"
                ],
            }
        else:
            estimate_renewable_caps = {"enable": False}
    else:
        estimate_renewable_caps = snakemake.config["electricity"][
            "estimate_renewable_capacities"
        ]
    if "enable" not in estimate_renewable_caps:
        logger.warning(
            "Missing key `enable` under config entry `estimate_renewable_capacities`. "
            "In future versions, this will raise an error. Falling back to False."
        )
        estimate_renewable_caps = {"enable": False}
    if "from_opsd" not in estimate_renewable_caps:
        logger.warning(
            "Missing key `from_opsd` under config entry `estimate_renewable_capacities`. "
            "In future versions, this will raise an error. "
            "Falling back to whether `renewable_capacities_from_opsd` is non-empty."
        )
        from_opsd = bool(
            snakemake.config["electricity"].get("renewable_capacities_from_opsd", False)
        )
        estimate_renewable_caps["from_opsd"] = from_opsd

    if estimate_renewable_caps["enable"]:
        if estimate_renewable_caps["from_opsd"]:
            tech_map = snakemake.config["electricity"]["estimate_renewable_capacities"][
                "technology_mapping"
            ]
            attach_OPSD_renewables(n, tech_map)
        estimate_renewable_capacities(n, snakemake.config)

    update_p_nom_max(n)

    add_nice_carrier_names(n, snakemake.config)

    n.meta = snakemake.config
    n.export_to_netcdf(snakemake.output[0])
