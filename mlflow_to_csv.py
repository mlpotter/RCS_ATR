import os
import re

import numpy as np
from mlflow.tracking import MlflowClient
import mlflow
import argparse

def get_experiment_df(experiment_name):
    client = MlflowClient()
    exp_id = client.get_experiment_by_name(experiment_name)

    if exp_id is None:
        return None
    runs = mlflow.search_runs(experiment_ids=[exp_id.experiment_id])

    runs.columns = [re.sub("params.", "", col) for col in runs.columns]
    runs.columns = [re.sub("metrics.", "", col) for col in runs.columns]

    runs["accuracy_time"] = ""
    runs["accuracy_single_time"] = ""

    for i,artifact in enumerate(runs["artifact_uri"]):
        try:
            runs.loc[i,"accuracy_time"] = mlflow.artifacts.load_text(artifact + "/accuracy_time.txt")
        except:
            pass
        try:
            runs.loc[i,"accuracy_single_time"] = mlflow.artifacts.load_text(artifact + "/accuracy_single_time.txt")
        except:
            pass
        try:
            runs.loc[i,"accuracy_mc_fuse"] = mlflow.artifacts.load_text(artifact + "/accuracy_mc_fuse.txt")
        except:
            pass
        try:
            runs.loc[i,"accuracy_mc_single"] = mlflow.artifacts.load_text(artifact + "/accuracy_mc_single.txt")
        except:
            pass

    return runs

def main(args):
    runs = get_experiment_df(args.experiment_name)
    print("Number of runs: ",runs.shape[0])
    runs.to_csv(os.path.join("results",args.experiment_name+".csv"),index=False)

if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='MLFLOW TO EXCEL',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--experiment_name', default="radar_target_recognition_joke",
                        help='mlflow experiment name')


    args = parser.parse_args()

    main(args)