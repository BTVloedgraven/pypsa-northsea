
from pathlib import Path
import shutil
import os
import jinja2
import filecmp
import pandas as pd

scenario_df = pd.DataFrame()

scenario_df["scenario1"] = {
    #Scenario 1
    "year": 2040,
    "lower": "greenfield",
    "upper": "nolim",
    "radial": "direct",
    "AC": 0,
    "DC": 0,
    "H2pipes": 0,
    "H2prod": 0,
    "costs": "balanced",
    "meshsize": 5000,
    "lin": "lin",
}
scenario_df["scenario2"] = {
    #Scenario 1
    "year": 2050,
    "lower": "greenfield",
    "upper": "nolim",
    "radial": "direct",
    "AC": 0,
    "DC": 0,
    "H2pipes": 0,
    "H2prod": 0,
    "costs": "balanced",
    "meshsize": 5000,
    "lin": "lin",
}
scenario_df["scenario3"] = {
    #Scenario 1
    "year": 2050,
    "lower": "eRisk2035",
    "upper": "nolim",
    "radial": "direct",
    "AC": 0,
    "DC": 0,
    "H2pipes": 0,
    "H2prod": 0,
    "costs": "balanced",
    "meshsize": 5000,
    "lin": "lin",
}
print(scenario_df["scenario1"].loc["meshsize"])

# elif id == 2:
#     #Scenario 2
#     year = 2050
#     lower = "greenfield"
#     upper = "nolim"
#     radial = "direct"
#     AC = 0
#     DC = 0
#     H2pipes = 0
#     H2prod = 0
#     costs = "balanced"
#     meshsize = 5000
#     lin = "lin"
# elif id == 3:
#     #Scenario 3
#     year = 2050
#     lower = "eRisk2035"
#     upper = "nolim"
#     radial = "direct"
#     AC = 0
#     DC = 0
#     H2pipes = 0
#     H2prod = 0
#     costs = "balanced"
#     meshsize = 5000
#     lin = "lin"
# elif id == 4:
#     #Scenario 4
#     year = 2050
#     lower = "eRisk2035"
#     upper = "onwind+p0.3"
#     radial = "direct"
#     AC = 0
#     DC = 0
#     H2pipes = 0
#     H2prod = 0
#     costs = "balanced"
#     meshsize = 5000
#     lin = "lin"
# elif id == 5:
#     #Scenario 5
#     year = 2050
#     lower = "eRisk2035"
#     upper = "onwind+p0.3"
#     radial = "off"
#     AC = 1
#     DC = 0
#     H2pipes = 0
#     H2prod = 0
#     costs = "balanced"
#     meshsize = 5000
#     lin = "lin"
# elif id == 6:
#     #Scenario 6
#     year = 2050
#     lower = "eRisk2035"
#     upper = "onwind+p0.3"
#     radial = "off"
#     AC = 1
#     DC = 1
#     H2pipes = 0
#     H2prod = 0
#     costs = "balanced"
#     meshsize = 5000
#     lin = "lin"
# elif id == 7:
#     #Scenario 7
#     year = 2050
#     lower = "eRisk2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 0
#     H2prod = 0
#     costs = "balanced"
#     meshsize = 5000
#     lin = "lin"
# elif id == 8:
#     #Scenario 8
#     year = 2050
#     lower = "eRisk2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "balanced"
#     meshsize = 5_000
#     lin = "lin"
# elif id == 9:
#     #Scenario 9
#     year = 2050
#     lower = "eRisk2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "balanced"
#     meshsize = 20_000
#     lin = "lin"



opts = "Co2L0.0-3H-CCL"
if upper != "nolim":
    opts += f"-{upper}"

if year == 2040:
    ll_wildcard = "v1.5"
    scaling_factor = 1.3
elif year == 2050:
    ll_wildcard = "v1.75"
    scaling_factor = 1.4

if id == 4:
    ll_wildcard = "copt"

def populate_config(infp: str, outfp: str, **kwargs):

    with open(infp) as infile:
        rendered_template = jinja2.Template(infile.read()).render(**kwargs)

        with open(outfp, "w") as outfile:
            outfile.writelines(rendered_template)

scenario = f"{year}_{lower}_{upper}_radial{radial}_AC{AC}_DC{DC}_H2pipes{H2pipes}_H2prod{H2prod}_{costs}_{meshsize}_{lin}"

prepared_path = "scenarios/prepared_scenarios" / Path(scenario)
solved_path = "scenarios/solved_scenarios" / Path(scenario)

populate_config(
    infp="scenarios/config_template.yaml",
    outfp="config.yaml",
    snapshot_year=year,
    ll_wildcard=ll_wildcard,
    opts=opts,
    radial=radial,
    AC=AC,
    DC=DC,
    H2pipes=H2pipes,
    H2prod=H2prod,
    scaling_factor=scaling_factor,
    meshsize=meshsize
)

os.system(f"mkdir {prepared_path}")

def copy_file(file, destination):
    if not os.path.exists(destination):
        shutil.copy(file, destination)
    elif not filecmp.cmp(file, destination):
        shutil.copy(file, destination)
    return

copy_file("staged_data" / Path(f"costs_{costs}.csv"), Path("data/costs.csv"))
copy_file("staged_data" / Path(f"agg_p_nom_minmax_{lower}.csv"), Path(f"{prepared_path}/agg_p_nom_minmax.csv"))
copy_file("staged_data" / Path(f"custom_bus_loads_{year}.csv"), Path("data/custom_bus_loads.csv"))
copy_file("staged_data" / Path(f"custom_national_loads_{year}.csv"), Path("data/custom_national_loads.csv"))
copy_file("staged_data" / Path(f"custom_powerplants_{year}.csv"), Path("data/custom_powerplants.csv"))
copy_file("staged_data" / Path(f"hydrogen_demand_per_bus_{year}.csv"), Path("data/hydrogen_demand_per_bus.csv"))
copy_file("staged_data" / Path(f"offshore_shapes_meshed_{meshsize}.geojson"), Path("resources/offshore_shapes_meshed.geojson"))

os.system(f"rm networks/elec_s_100_ec.nc")
os.system(f"rm networks/elec_s_100_ec_l{ll_wildcard}_{opts}.nc")
os.system(f"snakemake -j 1 networks/elec_s_100_ec_l{ll_wildcard}_{opts}.nc")

shutil.copy(f"networks/elec_s_100_ec_l{ll_wildcard}_{opts}.nc", prepared_path / f"elec_s_100_ec_l{ll_wildcard}_{opts}.nc")

# os.system(f"snakemake -j 1 {path}/solved_scenario/elec_s_100_ec_l{ll_wildcard}_{opts}.nc")

populate_config(
    infp="scenarios/solve_scenario_template.slurm",
    outfp=prepared_path / f"solve_{scenario}.slurm",
    id = id,
    outfile=solved_path / f"elec_s_100_ec_l{ll_wildcard}_{opts}.nc",
    plot_p_nom=solved_path / f"elec_s_100_ec_l{ll_wildcard}_{opts}_p_nom.png",
    plot_h2=solved_path / f"elec_s_100_ec_l{ll_wildcard}_{opts}_h2.png",
)

os.system(f"ssh s2182122@amhead.et.utwente.nl 'mkdir ~/pypsa-northsea/{prepared_path}'")
os.system(f"scp -r ./{prepared_path} s2182122@amhead.et.utwente.nl:~/pypsa-northsea/scenarios/prepared_scenarios")
