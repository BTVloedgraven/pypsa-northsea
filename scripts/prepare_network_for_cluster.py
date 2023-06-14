
from pathlib import Path
import shutil
import os
import jinja2
import filecmp

id = int(input("Please provide a scenario id: "))

# if id == 1:
#     #Scenario 1
#     year = 2040
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
#     lower = "eRisk+2035"
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
#     lower = "eRisk+2035"
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
#     lower = "eRisk+2035"
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
#     lower = "eRisk+2035"
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
#     lower = "eRisk+2035"
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
#     lower = "eRisk+2035"
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
#     lower = "eRisk+2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "balanced"
#     meshsize = 20_000
#     lin = "lin"
# elif id == 10:
#     #Scenario 10
#     year = 2050
#     lower = "eRisk+2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "highH2"
#     meshsize = 5_000
#     lin = "lin"
# elif id == 11:
#     #Scenario 11
#     year = 2050
#     lower = "eRisk+2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "highDC"
#     meshsize = 5_000
#     lin = "lin"
# elif id == 12:
#     #Scenario 12
#     year = 2050
#     lower = "eRisk+2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "highDC"
#     meshsize = "EEZ"
#     lin = "lin"
# elif id == 13:
#     #Scenario 13
#     year = 2050
#     lower = "eRisk+2035"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "highDC"
#     meshsize = "EEZ"
#     lin = "nonlin"
# elif id == 14:
#     #Scenario 14
#     year = 2050
#     lower = "300GW"
#     upper = "onwind+p0.3"
#     radial = "all"
#     AC = 1
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "balanced"
#     meshsize = 5_000
#     lin = "lin"
# elif id == 15:
#     #Scenario 15
#     year = 2050
#     lower = "eRisk+2035"
#     upper = "onwind+p0.3"
#     radial = "off"
#     AC = 0
#     DC = 1
#     H2pipes = 1
#     H2prod = 1
#     costs = "balanced"
#     meshsize = 20_000
#     lin = "lin"


def populate_config(infp: str, outfp: str, **kwargs):

    with open(infp) as infile:
        rendered_template = jinja2.Template(infile.read()).render(**kwargs)

        with open(outfp, "w") as outfile:
            outfile.writelines(rendered_template)

#add to options.csv
import pandas as pd
options_df = pd.read_csv('scenarios/options.csv', index_col=0)
scenario = f"scenario_{id}"
option = options_df.loc[scenario]

year = option['year']
lower = option['renewable_lower']
upper = option['upper']
radial = option['radial']
AC = option['AC']
DC = option['DC']
H2pipes = option['H2pipes']
H2prod = option['H2prod']
costs = option['costs']
meshsize = option['meshsize']
lin = option['lin']

if id in [12,13,21,22]:
    time_res = "24H"
else:
    time_res = "3H"

opts = f"Co2L0.0-{time_res}-CCL"

if upper != "nolim":
    opts += f"-{upper}"

if year == 2040:
    ll_wildcard = "v1.5"
    scaling_factor = 1.3
elif year == 2050:
    ll_wildcard = "v1.75"
    scaling_factor = 1.4

scenario = {
    "scenario": [f"scenario_{id}"],
    "year": [year],
    "renewable_lower": [lower],
    "upper": [upper],
    "radial": [radial],
    "AC": [AC],
    "DC": [DC],
    "H2pipes": [H2pipes],
    "H2prod": [H2prod],
    "costs": [costs],
    "meshsize": [meshsize],
    "lin": [lin],
}

prepared_path = f"scenarios/prepared_scenarios/scenario_{id}"
solved_path = f"scenarios/solved_scenarios/scenario_{id}"

os.system(f"mkdir {prepared_path}")

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

shutil.copy(f"networks/elec_s_100_ec_l{ll_wildcard}_{opts}.nc", f"{prepared_path}/prepared_scenario_{id}.nc")

populate_config(
    infp="scenarios/solve_scenario_template.slurm",
    outfp=f"{prepared_path}/solve_scenario_{id}.slurm",
    id = id,
    outfile=f"{solved_path}/solved_scenario_{id}.nc",
    plot_p_nom=f"{solved_path}/plot_scenario_{id}_p_nom.png",
    plot_h2=f"{solved_path}/plot_scenario_{id}_h2.png",
)
