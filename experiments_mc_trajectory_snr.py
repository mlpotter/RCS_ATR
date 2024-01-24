import os
import re

import numpy as np
from mlflow.tracking import MlflowClient
import mlflow
from subprocess import Popen
import pdb


def get_experiment_df(experiment_name):
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(experiment_name)

    if exp_id is None:
        return None
    runs = mlflow.search_runs(experiment_ids=[exp_id.experiment_id])
    columns = [col for col in runs.columns if "param" in col]
    runs = runs.loc[:, columns]
    runs.columns = [re.sub("params.", "", col) for col in columns]

    return runs

def get_args_dictionary(file):
    keyvals = [re.sub("--", "", word).split("=") for word in file.split(" ")]
    dictionary = {}
    for tuple in keyvals:
        if tuple[0] == "mlflow_track":
            dictionary["mlflow_track"] = "True"
        elif tuple[0] == "no-mlflow_track":
            dictionary["mlflow_track"] = "False"
        elif tuple[0] == "geometry":
            dictionary["geometry"] = "True"
        elif tuple[0] == "no-geometry":
            dictionary["geometry"] = "False"
        else:
            dictionary[tuple[0]] = tuple[1]
    return dictionary

def dictionary_equality(dict1,dict2):
    equality = [dict1[key] == item for key,item in dict2.items() if key in dict2]
    return np.all(equality)

blocking=False
batch_command = "--job-name=freedom --exclusive --cpus-per-task=18 --mem=20Gb --partition=short"

num_points = [10000]
n_radars = [4,16,64]
MC_trials = [10]

elevations = [(0.0,180.0)]
azimuths = [(90.0,180.0)]
angles = list(zip(azimuths,elevations))

model_choices = ["logistic","xgboost","mlp"]
snr_constraints = [-20.0,-10.0,0.0,10.0,20.0]

# noises = [("white","random"),("white","constant"),("color","random")]
noises = [("color","random")]
jitter_widths = [(0.0,0.0),(10.0,10.0),(20.0,20.0),(50.0,50.0),(80.0,80.0)]
# jitter_widths = [(0.0,0.0)]#,(10.0,10.0)]

# Trajectory Parameters
TN = 100
time_step_size = 0.1
vx = 50
yaw_range = "np.pi/15"
pitch_range = "np.pi/20"
roll_range = "0"

fusion_methods = ["average","fusion","max"]
experiment_name = "radar_target_recognition_snr_trajectory_geometry_avg_fixed"
random_seed = 123


geometry_use = "geometry"
mlflow_track = "mlflow_track"

previous_runs = get_experiment_df(experiment_name)

if __name__ == "__main__":
    print(experiment_name)
    os.makedirs("logs",exist_ok=True)

    for model_choice in model_choices:
        for fusion_method in fusion_methods:
            for num_pointsi in num_points:
                for MC_trialsi in MC_trials:
                    for n_radar in n_radars:
                        for snr_constraint in snr_constraints:
                            for angle in angles:

                                (azimuth_center,azimuth_spread),(elevation_center,elevation_spread) = angle

                                for noise_choices in noises:
                                    noise_color, noise_method = noise_choices

                                    for width in jitter_widths:
                                        azimuth_jitter_width,elevation_jitter_width = width

                                        file = f"--num_points={num_pointsi} "\
                                        f"--MC_Trials={MC_trialsi} "\
                                        f"--n_radars={n_radar} "\
                                        f"--TN={TN} "\
                                        f"--time_step_size={time_step_size} "\
                                        f"--vx={vx} "\
                                        f"--yaw_range={yaw_range} "\
                                        f"--pitch_range={pitch_range} "\
                                        f"--roll_range={roll_range} "\
                                        f"--noise_method={noise_method} "\
                                        f"--color={noise_color} "\
                                        f"--elevation_center={elevation_center} "\
                                        f"--elevation_spread={elevation_spread} "\
                                        f"--azimuth_center={azimuth_center} "\
                                        f"--azimuth_spread={azimuth_spread} "\
                                        f"--{geometry_use} "\
                                        f"--{mlflow_track} "\
                                        f"--single_method=random "\
                                        f"--SNR_constraint={snr_constraint} "\
                                        f"--azimuth_jitter_width={azimuth_jitter_width} "\
                                        f"--elevation_jitter_width={elevation_jitter_width} " \
                                        f"--azimuth_jitter_bounds=0_180 " \
                                        f"--elevation_jitter_bounds=-90_90 " \
                                        f"--model_choice={model_choice} "\
                                        f"--experiment_name={experiment_name} " \
                                        f"--fusion_method={fusion_method} "\
                                        f"--random_seed={random_seed}"

                                        if previous_runs is not None:
                                            previous_dict = get_args_dictionary(file)
                                            # pdb.set_trace()
                                            if previous_runs.apply(lambda x: x.to_dict() == get_args_dictionary(file),axis=1).sum() >= 1:
                                                print("Previous run found with same configuration...")
                                                continue

                                        print(file)
                                        if blocking:
                                            os.system(f"python main_mc_trajectory_SNR.py {file}")
                                        else:
                                            file_full = f"python main_mc_trajectory_SNR.py {file}"
                                            print(f"sbatch execute.bash '{file_full}'")
                                            Popen(f"sbatch execute.bash '{file_full}'",shell=True)
