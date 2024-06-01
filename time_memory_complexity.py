from sklearnex import patch_sklearn,config_context
patch_sklearn()



from src.data_loader import DRONE_RCS_CSV_TO_XARRAY,RCS_TO_DATASET_Single_Point,dataset_to_tensor,dataset_train_test_split
from src.trajectory_loader import RCS_TO_DATASET_Trajectory
from src.noise_generator import add_noise,generate_cov,add_jitter,add_noise_trajectory
from src.models import distributed_recursive_classifier
from src.misc import radar_grid
import xarray as xr
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay

import pickle

from xgboost import XGBClassifier

import os
import sys
import contextlib
from time import time

import argparse


# select which scikit learn ML model to use
# include standard normalization when appropriate
def select_model(model_choice,args):
    if model_choice == "logistic":
        model = LogisticRegression(n_jobs=-2,random_state=123)
    elif model_choice == "xgboost":
        model = XGBClassifier(n_jobs=-2,random_state=123)
    elif model_choice == "knn":
        model = KNeighborsClassifier(weights='distance',n_jobs=-2,random_state=123)
        return model
    elif model_choice == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(50,50,50),random_state=123)
    elif model_choice == "empty":
        return None
    else:
        sys.exit("MODEL CHOICE NOT VALID!")

    clf = make_pipeline(StandardScaler(),model)
    return clf

def main(args):
    # set the random seed for reproducibility
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # folder where the drone rcs data files (.csv files) are located...
    DRONE_RCS_FOLDER =  "Drone_RCS_Measurement_Dataset"

    # convert the csv data into xarrays (concatenate all the frequencies)
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER)

    # get the drone names and the number frequencies
    drone_names = list(drone_rcs_dictionary.keys())
    n_freq = len(drone_rcs_dictionary[drone_names[0]].coords["f[GHz]"])

    # specify where the target may be located with respect to world coordinate frame
    xlim = [-150, 150];  ylim = [-150, 150]; zlim = [200, 300]

    # create the bounding box to initialize target locations and radar locations on the z=0 plane
    bounding_box = np.array([xlim, ylim, zlim])


    # initialize empty result arrays for the mc trials
    results = np.zeros((args.MC_Trials,))
    results_single = np.zeros((args.MC_Trials,))
    results_mc_fuse = np.zeros((args.MC_Trials,))
    results_mc_single = np.zeros((args.MC_Trials,))

    # select the model with/without data standard normalization
    clf = select_model(args.model_choice,args)


    # generate unit trace covariance matrix for single radar scenario
    covs_single = generate_cov(TraceConstraint=1, d=n_freq, N=args.MC_Trials,
                                blocks=args.n_radars, color=args.color,
                                noise_method=args.noise_method)

    # iterate MC trials
    accuracy_time_fuse_avg = 0
    accuracy_time_single_avg = 0
    memory_avg = 0
    time_avg = 0

    for mc_trial in range(args.MC_Trials):
        print("\n","="*10,f"MC TRIAL {mc_trial}","="*10)

        # generate the uncorrupted RCS signals in the form of dictionary {"RCS":,"ys","azimuth":,"elevation":} for single radar
        dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                                     azimuth_center=args.azimuth_center, azimuth_spread=args.azimuth_spread,
                                                     elevation_center=args.elevation_center, elevation_spread=args.elevation_spread,
                                                     num_points=args.num_points,
                                                     method=args.single_method,random_seed=args.random_seed+10*mc_trial)

        # add gaussian noise to RCS at a fixed SNR value
        dataset_single["RCS"] = add_noise(dataset_single["RCS"],args.SNR_constraint,covs_single[mc_trial])
        # add the AZ/EL jitter noise to the data
        dataset_single["azimuth"] = add_jitter(dataset_single["azimuth"],args.azimuth_jitter_width,eval(args.azimuth_jitter_bounds.split("_")[0]),eval(args.azimuth_jitter_bounds.split("_")[1]))
        dataset_single["elevation"] = add_jitter(dataset_single["elevation"],args.elevation_jitter_width,eval(args.elevation_jitter_bounds.split("_")[0]),eval(args.elevation_jitter_bounds.split("_")[1]))

        # split into train-test data dictionaries
        # dataset_train,dataset_test = dataset_train_test_split(dataset_single)

        # convert the dataset dictionaries into numpy arrays with the azimuth and elevation information
        X_train,y_train = dataset_to_tensor(dataset_single,args.geometry)
        # X_test,y_test = dataset_to_tensor(dataset_test,args.geometry)

        # fit single radar CLF
        time_train = time()
        clf.fit(X_train, y_train.ravel())
        time_train = time() - time_train
        print("Train Time: ",time_train)

        # Pickle the estimator without compression
        with open(os.path.join('results',f'estimator{mc_trial}.pkl'), 'wb') as file:
            pickle.dump(clf, file)

        clf_memory = os.path.getsize(os.path.join('results',f'estimator{mc_trial}.pkl'))
        print("Model Size: ",clf_memory)
        # memory_size = deep_sizeof(clf, with_overhead=True, verbose=True)
        # print(memory_size)
        # get the accuracy of a single radar on balanced dataset
        # y_pred = clf.predict(X_test)
        # accuracy_single = accuracy_score(y_test.ravel(), y_pred.ravel())
        # results_single[mc_trial] = accuracy_single

        # simulate random radar positions for args.n_radars on the z=0 plane
        radars = radar_grid(n_radars=args.n_radars, xlim=xlim, ylim=ylim)

        # generate the multi radar dataset dictionary {"RCS":,"ys","azimuth":,"elevation":}
        dataset_multi = RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                                  time_step_size=args.time_step_size, vx=args.vx,
                                                  yaw_range=eval(args.yaw_range),
                                                  pitch_range=eval(args.pitch_range),
                                                  roll_range=eval(args.roll_range),
                                                  bounding_box=bounding_box,
                                                  TN=args.TN, radars=radars,
                                                  num_points=100,random_seed=args.random_seed+10*mc_trial,#X_test.shape[0],
                                                  verbose=False)

        # add gaussian noise to RCS at a fixed SNR value.. Note we use a block diagonal matrix (so we assume each radar measure is independent)
        dataset_multi["RCS"] = add_noise_trajectory(dataset_multi["RCS"], args.SNR_constraint, covs_single[mc_trial])

        # add the AZ/EL jitter noise to the data
        dataset_multi["azimuth"] = add_jitter(dataset_multi["azimuth"],args.azimuth_jitter_width,eval(args.azimuth_jitter_bounds.split("_")[0]),eval(args.azimuth_jitter_bounds.split("_")[1]))
        dataset_multi["elevation"] = add_jitter(dataset_multi["elevation"],args.elevation_jitter_width,eval(args.elevation_jitter_bounds.split("_")[0]),eval(args.elevation_jitter_bounds.split("_")[1]))

        drc = distributed_recursive_classifier(dataset_multi["n_classes"], use_geometry=args.geometry)

        _,y_test = dataset_to_tensor(dataset_multi,args.geometry)

        time_test = time()
        y_pred,y_pred_history = drc.predict(clf, dataset_multi, fusion_method=args.fusion_method)
        time_test = time()-time_test
        print("Whole Dataset Test Time: ",time_test)

        accuracy_distributed = accuracy_score(y_test.ravel(), y_pred.argmax(-1).ravel())
        results[mc_trial] = accuracy_distributed
        results_mc_fuse[mc_trial] = accuracy_distributed

        dataset_multi["n_radars"] = 1
        y_pred_1, y_pred_history_1 = drc.predict(clf, dataset_multi,fusion_method="average")
        accuracy_single = accuracy_score(y_test.ravel(), y_pred_1.argmax(-1).ravel())
        results_single[mc_trial] = accuracy_single
        results_mc_single[mc_trial] = accuracy_single

        print("MC Trials={} Accuracy={:.3f} Accuracy Single={:.3f}".format(mc_trial,accuracy_distributed,accuracy_single))

        # time the distributed recursive classifier prediction time on single instance

        n_traj = dataset_multi["RCS"].shape[0]
        time_dataset = 0
        for i in range(n_traj):
            dataset_individual =  {"RCS":dataset_multi["RCS"][[i]],
                                   "azimuth":dataset_multi["azimuth"][[i]],
                                    "elevation":dataset_multi["elevation"][[i]],
                                    "rho":dataset_multi["rho"][[i]],
                                    "ys": dataset_multi["ys"][[i]],
                                    "n_radars":dataset_multi["n_radars"],
                                    "n_freq":dataset_multi["n_freq"],
                                    "yaws":dataset_multi["yaws"][[i]],
                                    "pitchs":dataset_multi["pitchs"][[i]],
                                    "rolls":dataset_multi["rolls"][[i]],
                                    "translations":dataset_multi["translations"][[i]],
                                    "TN":dataset_multi["TN"],
                                    "time_step_size":dataset_multi["time_step_size"],
                                    "n_classes":dataset_multi["n_classes"]}
            time_test = time()
            y_pred,y_pred_history = drc.predict(clf, dataset_individual, fusion_method=args.fusion_method)
            time_test = time()-time_test
            time_dataset += time_test

        time_dataset = time_dataset / n_traj
        print("MC Trials={} Inference Time={:.7f}".format(mc_trial,time_dataset))


        accuracy_time_fuse = (y_pred_history.argmax(-1) == dataset_multi["ys"]).mean(0)
        accuracy_time_single = (y_pred_history_1.argmax(-1) == dataset_multi["ys"]).mean(0)

        accuracy_time_fuse_avg += accuracy_time_fuse
        accuracy_time_single_avg += accuracy_time_single
        memory_avg += clf_memory
        time_avg += time_dataset


    print()
    print("="*10)
    print("Multi Radar Accuracy:", results.mean())
    print("Single Radar Accuracy:", results_single.mean())
    print("Average model memory complexity:",memory_avg/args.MC_Trials)
    print("Average model time complexity:",time_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Radar Cross Section Experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Experiment Parameters
    parser.add_argument('--num_points',default=1000,type=int, help='Number of random points per class label')
    parser.add_argument('--MC_Trials',default=10,type=int, help='Number of MC Trials')
    parser.add_argument('--n_radars', default=16,type=int, help='Number of radars in the xy grid')
    parser.add_argument('--geometry', action=argparse.BooleanOptionalAction,default=False,help='Do you want az-el measurements in experiments? --geometry for yes --no-geometry for no')
    parser.add_argument("--fusion_method",type=str,default="average",help="how to aggregate predictions of distributed classifier")
    parser.add_argument("--model_choice",type=str,default="xgboost",help="Model to train")

    # Drone PARAMETERS / Trajectory Parameters
    parser.add_argument('--TN', default=20,type=int, help='Number of time steps for the experiment')
    parser.add_argument('--vx', default=50,type=int, help='The Velocity of the Drone in the forward direction')
    parser.add_argument('--yaw_range', default="np.pi/8",type=str, help='The yaw std of the random drone walk')
    parser.add_argument('--pitch_range', default="np.pi/15",type=str, help='The pitch std of the random drone walk')
    parser.add_argument('--roll_range', default="0",type=str, help='The roll std of the random drone walk')
    parser.add_argument('--time_step_size', default=0.1,type=float, help='The Velocity of the Drone in the forward direction')

    # RCS NOISE PARAMETERS
    parser.add_argument('--noise_method', type=str,default="random", help='The noise method to choose from (see code for examples)')
    parser.add_argument('--color', type=str,default="white", help='The type of noise (color,white)')
    parser.add_argument('--SNR_constraint',type=float,default=0,help="The signal to noise ratio")
    parser.add_argument('--azimuth_center', default=90,type=float, help='azimuth center to sample for target single point RCS, bound box for random')
    parser.add_argument('--azimuth_spread', default=5,type=float, help='std of sample for target single point RCS, bound box for random')
    parser.add_argument('--elevation_center', default=0,type=float, help='elevation center to sample for target single point RCS, bound box for random')
    parser.add_argument('--elevation_spread', default=5,type=float, help='std of sample for target single point RCS, bound box for random')
    parser.add_argument('--single_method', type=str,default="target", help='target is for sampling around some designated point, random is to sample uniformly over grid')

    # Azimuth and Elevation Jitter Parameters
    parser.add_argument('--azimuth_jitter_width', default=10,type=float, help='the width of the jitter for azimuth')
    parser.add_argument('--azimuth_jitter_bounds', default="0_180",type=str, help='lower and upper bound of the azimuth when adding noise to clip')
    parser.add_argument('--elevation_jitter_width', default=10,type=float, help='the width of the jitter for elevation')
    parser.add_argument('--elevation_jitter_bounds', default="-90_90",type=str, help='lower and upper bound of the elevation when adding noise to clip')

    # Model Tracking parameters
    parser.add_argument("--random_seed",type=int,default=123)

    args = parser.parse_args()

    experiment_name = f"experiment=num_points={args.num_points} n_radars={args.n_radars} " \
                      f"color={args.color} noise_method={args.noise_method} MC_trials={args.MC_Trials}" + ".xlsx"

    from datetime import datetime
    from pytz import timezone

    tz = timezone('EST')
    print(datetime.now(tz))

    print(f"Model Choice = {args.model_choice}")
    print(f"Number of Radars = {args.n_radars}")
    print(f"Noise Method = {args.noise_method} Noise Color={args.color}")
    print(f"Fusion Method = {args.fusion_method}")
    print(f"SNR = {args.SNR_constraint}")
    print(f"Geometry = {args.geometry}")
    print(f"Az/El Jitter = {args.azimuth_jitter_width},{args.elevation_jitter_width}")
    print(f"Az/El Bounds = {args.azimuth_jitter_bounds},{args.elevation_jitter_bounds}")

    os.makedirs(os.path.join("results","temp"),exist_ok=True)

    main(args)