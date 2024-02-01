import numpy as np
import random
import pandas as pd

from copy import deepcopy

import pandas as pd
import glob

import xarray as xr
import matplotlib.pyplot as plt
import os

from src.geometry import calculate_3d_angles_ref
from tqdm import tqdm
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import random

# TODO: WRITE CODE TO SAVE, LOAD PICKLE FILES
# TODO: note that parrot_HH and MAVIC_HH have elevation from 0-180... is this mistake?
def DRONE_RCS_CSV_TO_XARRAY(path,visualize=False,verbose=False,exponentiate=False):
    """
    @param path: The filepath to the folder which contains the .csv files with the elevation,angle,rcs data for each drone
    @param visualize: visualize=True generates the RCS[dB] histogram for each frequency of a drone
    @param verbose: verbose=True prints out debugging statements
    @param exponentiate: exponentiate=True converts RCS[dB] to RCS[m^2]
    @return: drone_dictionary {drone name : RCS xarray of shape N Frequency x N azimuth x N elevation}, label_dictionary {class number : drone name}
    """


    if verbose:
        print("LOAD RCS DATA FROM CSV FILES")
    # create a list object that contains all the .csv files with the drones elevation,azimuth, and RCS values
    csv_files = glob.glob(os.path.join(path,"*.csv"))

    # initialize drone dictionary {"drone name": rcs xarray stacked over frequency}
    drone_dictionary = {}

    # iterate through each drone RCS csv file
    for file_path in csv_files:

        # create a pandas dataframe by reading the csv file with columns frequency, elevation, azimuth, and RCS
        data = pd.read_csv(file_path)

        # make sure consistent organization amongst data csv files...
        # sort the data in ascending order and reset the index of the row
        data = data.sort_values(by=["f[GHz]", "phi[deg]", "theta[deg]"], ascending=True).reset_index(drop=True)

        # extract the name of the drone from the .csv filename
        drone_name = file_path.split(os.sep)[-1].replace(".csv", "")

        # visualize the histograms the RCS values per each frequency
        if visualize:
            data["RCS[dB]"].hist(by=data["f[GHz]"], figsize=(30, 30))
            plt.suptitle(drone_name)
            plt.show()

        # we skip these drones in our analysis
        if drone_name in ["Mavic_HH","Parrot_HH","battery_HH"]:
            continue

        # we only focus on the HH polarization of the transmitted and received electromagnetic wave
        if "HH" not in drone_name:
            continue

        # print summary statistics of the min and max frequency, elevation, azimuth, and rcs values
        if verbose:
            descriptive_df = pd.concat((pd.DataFrame(data.min(0)).T, pd.DataFrame(data.max(0)).T))
            descriptive_df.index = ["min", "max"]
            print(f"Loaded {drone_name}")
            print(descriptive_df.to_string())

        # Assuming 'f[GHz]' are categories you want to filter on to extract the data corresponding to specific frequency
        unique_frequencies = data['f[GHz]'].unique()

        # list of RCS xarray objects, where RCS xarray are for each frequency and azimuth (row) - elevation (column) arrays
        RCS_arrays = []

        # iterate through each unique frequency in the data analysis
        for i, frequency in enumerate(unique_frequencies, 1):

            # filter the frequency,elevation, azimuth, and rcs values corresponding to the specific frequency of a drone
            specific_data = data[data['f[GHz]'] == frequency]

            # Extracting specific data columns for plotting
            # theta is elevation, phi is azimuth
            theta = specific_data['theta[deg]']
            phi = specific_data['phi[deg]']
            rcs = specific_data['RCS[dB]']  # color coding based on RCS values

            # convert the RCS [dB] into RCS[ m^2]
            if exponentiate:
                rcs = 10**(rcs/10)

            # print("@{}GHz Min RCS {:0.3f} , Max RCS {:.3f}".format(frequency,rcs.min(), rcs.max()))

            # convert dataframe to xarray, where x coordinates are azimuth and y coordinates are "elevation"
            coords = {"azimuth": phi.unique(), "elevation": theta.unique()}

            n_azimuth = len(phi.unique())
            n_elevation = len(theta.unique())

            # create an named xarray, where the row is the azimuth and the column is the elevation
            # N azimuth x N elevation
            RCS_array = xr.DataArray(rcs.values.reshape(n_azimuth,n_elevation), coords=coords, dims=["azimuth", "elevation"])

            # name the RCS_array based on the frequency the subset of data is for
            RCS_array.name = f"{frequency}[GHz]"

            # add the RCS_array with dimensions 1 x N azimuth x N elevation
            RCS_arrays.append(RCS_array.expand_dims("f[GHz]"))

        # add the concatenated xarray to the drone dictionary
        # concatenated xarray is N Frequency x N Azimuth x N Elevation
        drone_dictionary[drone_name] = xr.concat(RCS_arrays, pd.Index(unique_frequencies,name="f[GHz]"))

        # debugging statement which checks the shape of the xarray corresponding to a drone
        if verbose:
            print("RCS shape: ",drone_dictionary[drone_name].shape)

    # create an sklearn LabelEncoder which maps the drone name to a class number
    label_dictionary = LabelEncoder().fit(list(drone_dictionary.keys()))

    print(list(zip(label_dictionary.transform(list(drone_dictionary.keys())),list(drone_dictionary.keys()))))

    # create a dictionary {class number : drone name}
    drone_dictionary = {label_dictionary.transform([k])[0]:v for k,v in drone_dictionary.items()}

    return drone_dictionary,label_dictionary

def simulate_target_azim_elev(radars,yaw_lim,pitch_lim,roll_lim,bounding_box,num_points):
    """
    @param radars: a numpy array [[x1, y1, z1], [x2, y2, z2], ..., [xJ, yJ , zJ]] denoting the radar locations
    @param yaw_lim: a list [lb ub] that contains the upper and lower bounds for uniform distribution samples for yaw values
    @param pitch_lim: a list [lb ub] that contains the upper and lower bounds for uniform distribution samples for pitch values
    @param roll_lim: a list [lb ub] that contains the upper and lower bounds for uniform distribution samples for roll values
    @param bounding_box: a numpy array [[x lb, x ub], [y lb, y ub], [z lb, z ub]] denoting the "box" which we sample uav locations
    @param num_points: the number of (azimuth, elevation, rcs) samples to generate from mapping yaw,pitch,roll,translation samples to azimuth,elevation,rcs
    @return: azimuth,elevation which are Number of samples x Number of Radars
    """

    # sample uniformly at random yaws, pitchs, and rolls within the lower and upper bounds specified
    # Number of samples x ,
    yaws = np.random.uniform(yaw_lim[0], yaw_lim[1], num_points)
    pitchs = np.random.uniform(pitch_lim[0], pitch_lim[1], num_points)
    rolls = np.random.uniform(roll_lim[0], roll_lim[1], num_points)

    # sample uniformly at random translations within the lower and upper bounds specified of the box
    # Number of samples x Dimension of Coordinates (in this case 3, XYZ)
    translations = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], num_points),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], num_points),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], num_points)
    ))

    # Number of samples x Number of Radars
    # rho is the distance between target and radar
    # azimuth and the xy angle from target to radar
    # elevation is the  angle between the z-axis and the lines from the target to radar
    range_, rho, azimuth, elevation = calculate_3d_angles_ref(translations, yaws, pitchs, rolls, radars)

    # convert everything from radians to degrees
    elevation = elevation * 180/np.pi; azimuth = azimuth * 180/np.pi;

    # how to account for the negative elevations... what does a negative elevation even mean?

    # when we have negative azimuth, it means that it is flipped because symetry... (wrapping effect)
    # the target is symmetric along the z-plane (like a box)
    azimuth[azimuth < 0]  = (azimuth[azimuth < 0] + 180) % 360

    return azimuth,elevation

# TODO: Incorperate Dataset metadata to make dictionary data object ... such as the radars positions, yaws, bounding box, etc
def RCS_TO_DATASET(RCS_xarray_dictionary,radars,yaw_lim,pitch_lim,roll_lim,bounding_box,num_points,seed=123,verbose=False):
    """
    @param RCS_xarray_dictionary: a dictionary {drone name : rcs xarray stacked by frequency} . xarray is shape N Freq x N Azimuth x N Elevation
    @param radars: a numpy array [[x1, y1, z1], [x2, y2, z2], ..., [xJ, yJ , zJ]] denoting the radar locations
    @param yaw_lim: a list [lb ub] that contains the upper and lower bounds for uniform distribution samples for yaw values
    @param pitch_lim: a list [lb ub] that contains the upper and lower bounds for uniform distribution samples for pitch values
    @param roll_lim: a list [lb ub] that contains the upper and lower bounds for uniform distribution samples for roll values
    @param bounding_box: a numpy array [[x lb, x ub], [y lb, y ub], [z lb, z ub]] denoting the "box" which we sample uav locations
    @param num_points: the number of (azimuth, elevation, rcs) samples to generate from mapping yaw,pitch,roll,translation samples to azimuth,elevation,rcs
    @param seed: the random seed which allows reproducibility of data generation
    @param verbose: verbose=True for debugging output
    @return: a dataset dictionary  {"RCS":RCSs,"azimuth":azimuths,"elevation":elevations,"ys":ys,"n_radars":N_radars,"n_freq":N_freqs}.
    RCS is Number of samples x (Number of radars * Number of Frequency). Azimuth is Number of Samples x Number of Radars
    """

    # set the random seed for numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # the total number of radars in analysis
    N_radars = radars.shape[0]

    # initialize empy lists for RCS, class label, azimuth, and elevation respectively
    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []

    # initialize dictionary to keep track of the required number of samples left until we reach the goal dataset size
    # {drone class number : number of remaining samples required}
    drone_sample_count = {drone_key:num_points for drone_key in RCS_xarray_dictionary.keys()}


    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO X,y DATASET")

    start_time = time()

    # while there remains a drone class that still requires samples
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts>0])

        # generate the azimuth and elevations of the target with respect to radar line of sight.
        # number of samples x number of radars
        azimuth, elevation = simulate_target_azim_elev(radars, yaw_lim, pitch_lim, roll_lim, bounding_box,num_points)

        # iterate through each drone's RCS "stack" xarray
        for drone,RCS_array in RCS_xarray_dictionary.items():
            # print(drone)

            # check if this drone has enough samples already... if it does skip it
            if drone_sample_count[drone] <= 0:
                continue

            # all the valid azimuth and elevations for particular RCS_array
            azimuth_axes = RCS_array.coords["azimuth"].values
            elevation_axes = RCS_array.coords["elevation"].values
            frequency_axes = RCS_array.coords["f[GHz]"].values

            # the number of unique frequencies in the xarray stack
            N_freqs = len(frequency_axes)

            # boundary of azimuth and elevations to check for valid sample generated
            valid_azimuths, valid_elevations = [0,180],[-90,90]#[np.min(azimuth_axes),np.max(azimuth_axes)],[np.min(elevation_axes),np.max(elevation_axes)]

            # the random azimuth and elevation MUST BE WITHIN RANGE OF THE REAL DATA!
            # a sample is only valid if ALL the azimuths and elevations wrt each radar are VALID

            # Number of samples x Number of Radars
            valid_sample_idx = ((azimuth <= valid_azimuths[1]) & (azimuth >= valid_azimuths[0]) & (elevation <= valid_elevations[1]) & (elevation >= valid_elevations[0]))

            # Check if the generated azimuths and elevations (per sample) are valid for all radars.
            # Number of samples
            valid_sample_idx = valid_sample_idx.all(-1)

            # subset only the valid azimuths and elevations to create RCS data
            # Number of samples x Number of Radars
            azimuth_copy = azimuth[valid_sample_idx,:]
            elevation_copy = elevation[valid_sample_idx,:]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            # 1x1 array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            # number of frequencies x (number of samples * number of radars)
            try:
                # index (and interpolate if required) the RCS xarray stack across all frequencies for each simulated azimuth and elevation
                # number of frequencies x (number of samples * number of radars)
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))


            except:
                continue

            # reshape the xarray RCS_indexed (Number of Frequencies x (Number of Samples * Number of Radars) ) and convert to numpy array such that ...
            # number of samples x (number of radars * number of frequencies)
            RCS_indexed = RCS_indexed.values.T.reshape(-1,N_freqs*N_radars)

            #    -- s1 --  -- s2 --
            # f1 [r1 r2 r3 r1 r2 r3]
            # f2 [r1 r2 r3 r1 r2 r3]
            # f3 [r1 r2 r3 r1 r2 r3]
            # ..
            # fF [r1 r2 r3 r1 r2 r3]

            # tranpose gives
            #     f1 f2 f3 f4 ..    fF
            # |  [r1 r1 r1 r1 .. r1 r1]
            # s1 [r2 r2 r2 r2 .. r2 r2]
            # |  [r3 r3 r3 r3 .. r3 r3]
            # ..
            #    [r3 r3 r3 r3 .. r3 r3]

            # reshape gives
            #      f1 ... fF f1 ... fF f1 ... fF
            # s1  [r1 ... r1 r2 ... r2 r3 ... r3]
            # s2  [r1 ... r1 r2 ... r2 r3 ... r3]
            # s3  [r1 ... r1 r2 ... r2 r3 ... r3]
            # ..
            # sN  [r1 ... r1 r2 ... r2 r3 ... r3]

            # append to RCS,label, azimuth, and elevation list (add the samples to the current running dataset
            RCSs.append(RCS_indexed)
            ys.append(np.ones((valid_sample_idx.sum(),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)

            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - np.sum(valid_sample_idx))

        if verbose:
            print(drone_sample_count)

    # concetate all the data samples along the rows such that
    # RCSs is Number of samples x (Number of radars * Number of Frequencies))
    RCSs = np.vstack(RCSs)

    # Number of samples x Number of radars
    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)

    # Number of samples x 1
    ys = np.vstack(ys)

    # create a dataset dictionary to keep all generated data organized
    dataset = {
        "RCS":RCSs,
                "azimuth":azimuths,
                "elevation":elevations,
                "ys":ys,
                "n_radars":N_radars,
                "n_freq":N_freqs
            }

    end_time = time()
    if verbose:
        print("Dataset Creation Time: {:.3f}".format(end_time-start_time))

    return dataset


# TODO: note that parrot_HH and MAVIC_HH have elevation from 0-180... so elevation center and azimuth center will get screwy from user specified!
def RCS_TO_DATASET_Single_Point(RCS_xarray_dictionary,azimuth_center,azimuth_spread,elevation_center,elevation_spread,num_points,method="target",random_seed=123,verbose=False):
    """
    @param RCS_xarray_dictionary: a dictionary {drone name : rcs xarray stacked by frequency} . xarray is shape N Freq x N Azimuth x N Elevation
    @param azimuth_center: a float value see. method argument
    @param azimuth_spread: a float value see. method argument
    @param elevation_center: a float value. see method argument
    @param elevation_spread: a float value. see method argument
    @param num_points: the number of (azimuth, elevation, rcs) samples to generate from mapping yaw,pitch,roll,translation samples to azimuth,elevation,rcs
    @param method: target - sample from Gaussian distribution with mean of center and std spread (all in degrees). Random - sample from uniform distribution where [lb,ub] = [center-spread/2,center+spread/2]
    @param seed: the random seed which allows reproducibility of data generation
    @param verbose: verbose=True for debugging output
    @return: a dataset dictionary  {"RCS":RCSs,"azimuth":azimuths,"elevation":elevations,"ys":ys,"n_radars":1,"n_freq":N_freqs}.
    """

    # set the random seed for data generation reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    # dictionary to convert label to digit label
    # we only consider the perspective of a single radar...
    N_radars = 1

    # initialize empy lists to save generated data
    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []

    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO X,y DATASET")

    # initialize dictionary to keep track of the required number of samples left until we reach the goal dataset size
    # {drone class number : number of remaining samples required}
    drone_sample_count = {drone_key: num_points for drone_key in RCS_xarray_dictionary.keys()}
    start_time = time()

    # while there remains a drone class that still requires samples
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts > 0])

        if method == "target":
            # generate the azimuth and elevations of the target with respect to radar line of sight.
            # sample from a Gaussian distribution with mean center and std spread
            # number of points x 1
            azimuth = azimuth_spread * np.random.randn(num_points, 1) + azimuth_center

            # since we only have azimuth from 0-180, make sure to account for symetry of drone
            azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180) % 360

            # number of points x 1
            elevation = elevation_spread * np.random.randn(num_points, 1) + elevation_center

        elif method == "random":
            # define the lower and upper bound of the uniform distribution as [center-spread/2,center+spread/2]
            half_spread_az = azimuth_spread/2
            half_spread_el = elevation_spread/2

            # sample uniformly at random between the [lb,ub] specified above
            # number of points x 1
            azimuth = np.random.uniform(azimuth_center-half_spread_az,azimuth_center+half_spread_az,size=(num_points,1))
            elevation = np.random.uniform(elevation_center-half_spread_el,elevation_center+half_spread_el,size=(num_points,1))

            # since we only have azimuth from 0-180, make sure to account for symetry of drone
            azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180)

        # the method must be available
        else:
            raise Exception("Sorry, this method is not valid")

        # iterate through each drone's RCS "stack"
        for drone, RCS_array in RCS_xarray_dictionary.items():
            # print(drone)

            # check if this drone has enough samples already... if it does skip it
            if drone_sample_count[drone] <= 0:
                continue

            # all the valid azimuth and elevationns ffor particular RCS_array
            azimuth_axes = RCS_array.coords["azimuth"].values
            elevation_axes = RCS_array.coords["elevation"].values
            frequency_axes = RCS_array.coords["f[GHz]"].values

            N_freqs = len(frequency_axes)

            # boundary of azimuth and elevations to check for valid sample generated
            valid_azimuths, valid_elevations = [np.min(azimuth_axes), np.max(azimuth_axes)], [np.min(elevation_axes),
                                                                                              np.max(elevation_axes)]

            # the random azimuth and elevation MUST BE WITHIN RANGE OF THE REAL DATA!
            # a sample is only valid if ALL the azimuths and elevations wrt each radar are VALID
            # Number of samples x 1
            valid_sample_idx = ((azimuth <= valid_azimuths[1]) & (azimuth >= valid_azimuths[0]) & (
                        elevation <= valid_elevations[1]) & (elevation >= valid_elevations[0]))

            # Check if the generated azimuths and elevations (per sample) are valid for all radars (1 radar).
            # Number of samples
            valid_sample_idx = valid_sample_idx.all(-1)

            # if there are no valid generated samples, move on to the next drone
            if valid_sample_idx.sum() == 0:
                continue

            # subset only the valid azimuths and elevations to create RCS data
            # Number of samples x 1
            azimuth_copy = azimuth[valid_sample_idx, :]
            elevation_copy = elevation[valid_sample_idx, :]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            # 1x1 array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            try:
                # index (and interpolate if required) the RCS xarray stack across all frequencies for each simulated azimuth and elevation
                # number of frequencies x number of samples
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))


            except:
                continue

            RCS_indexed = RCS_indexed.values.T.reshape(-1,N_freqs*N_radars)
            #    -- s1 --  -- s2 --
            # f1 [r1       r1  ]
            # f2 [r1       r1  ]
            # f3 [r1       r1  ]
            # ..
            # fF [r1       r1  ]

            # tranpose gives
            #     f1 f2 f3 f4 ..   fF
            # s1  [r1 r1 r1 r1 ..  r1]
            # s2  [r1 r1 r1 r1 ..  r1]
            # s3  [r1 r1 r1 r1 ..  r1]
            # ..
            # sN  [r1 r1 r1 r1 ..  r1]

            # reshape gives
            #      f1 ... fF
            # s1  [r1 ... r1 ]
            # s2  [r1 ... r1 ]
            # s3  [r1 ... r1 ]
            # ..
            # sN  [r1 ... r1 ]


            # append to RCS,label, azimuth, and elevation list (add the samples to the current running dataset
            RCSs.append(RCS_indexed)
            ys.append(np.ones((valid_sample_idx.sum(),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)

            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - np.sum(valid_sample_idx))

    # concetate all the data samples along the rows such that
    # RCSs is Number of samples x Number of Frequencies
    RCSs = np.vstack(RCSs)

    # Number of samples x 1
    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)
    ys = np.vstack(ys)

    # create a dataset dictionary to keep all generated data organized
    dataset = {
        "RCS":RCSs,
                "azimuth":azimuths,
                "elevation":elevations,
                "ys":ys,
                "n_radars":1,
                "n_freq":N_freqs
            }

    end_time = time()
    # if verbose:
    print("Single Point Dataset Creation Time: {:.3f}".format(end_time-start_time))

    return dataset

def dataset_to_tensor(dataset,use_geometry):
    """
    @param dataset: a dataset dictionary  {"RCS":RCSs,"azimuth":azimuths,"elevation":elevations,"ys":ys,"n_radars":N_radars,"n_freq":N_freqs,...}.
    @param use_geometry: bool True: include azimuth and elevation in returned numpy array
    @return: Dataset as numpy array (Number of radars * Number of Frequency + Number of Radars + Number of Radars) and class labels as numpy array (number of samples x 1)
    """
    # Array of RCS values: Number of samples x (Number of radars * Number of Frequency)
    RCS = dataset["RCS"]

    X = RCS

    # concatenate the azimuth and elevation data to the RCS data
    # Number of Samples x (Number of radars * Number of Frequency + Number of Radars + Number of Radars)
    if use_geometry:
        azimuth = dataset["azimuth"]
        elevation = dataset["elevation"]
        X = np.concatenate((X,azimuth,elevation),axis=-1)

    # Array of class labels: Number of samples x 1
    y = dataset["ys"]
    return X,y

def dataset_train_test_split(dataset,test_size=0.2):
    """
    @param dataset: a dataset dictionary  {"RCS":RCSs,"azimuth":azimuths,"elevation":elevations,"ys":ys,"n_radars":N_radars,"n_freq":N_freqs,...}.
    @param test_size: the percentage of datapoints that should become the test dataset
    @return: train dataset and test dataset in the dataset dictionary format described in the dataset parameter
    """

    # the number of radars in the experiment
    n_radars = dataset["n_radars"]

    # the number of unique frequences in the expperiment
    n_freq = dataset["n_freq"]

    # numpy array of the RCS data of shape Number of Samples x (Number of radars * Number of Frequencies)
    RCS = dataset["RCS"]

    # numpy arrays of azimuth and elevation of shape Number of Samples x Number of radars
    azimuth = dataset["azimuth"]
    elevation = dataset["elevation"]

    # numpy array of class label of shape Number of Samples x 1
    y = dataset["ys"]

    # stratified partitioning of rcs array, azimuth array, elevation array, and class label array using scikit learn
    RCS_train, RCS_test, Az_train,Az_test,El_train,El_test, y_train, y_test = train_test_split(RCS,azimuth,elevation, y, test_size=test_size, random_state=123, stratify=y.ravel())

    # group all the train numpy arrays into a dataset dictionary object
    dataset_train = {
                "RCS":RCS_train,
                "azimuth":Az_train,
                "elevation":El_train,
                "ys":y_train,
                "n_radars":n_radars,
                "n_freq":n_freq
            }

    # group all the test numpy arrays into a dataset dictionary object
    dataset_test = {
                "RCS":RCS_test,
                "azimuth":Az_test,
                "elevation":El_test,
                "ys":y_test,
                "n_radars":n_radars,
                "n_freq":n_freq
            }

    return dataset_train,dataset_test

def visualize_RCS_dictionary(drone_rcs_dictionary,smooth_images=False,save_images=False):

    for drone_name, RCS_array in drone_rcs_dictionary.items():
        # Assuming 'f[GHz]' are categories you want to filter on for each subplot

        # Creating 4x2 subplots
        # hardcoded.. should dynamically change
        fig, axs = plt.subplots(3, 5, figsize=(31, 16))  # you can adjust the size as per your requirement

        # Flatten the 4x2 array to easily iterate over it
        axs = axs.flatten()

        unique_frequencies = RCS_array.coords["f[GHz]"].values
        print(f"{drone_name}")
        for i, (frequency, ax) in enumerate(zip(unique_frequencies, axs), 1):
            # Filter data for each unique 'f[GHz]'
            RCS_frequency = RCS_array.loc[frequency,:,:]
            RCS_min, RCS_max = RCS_frequency.min(), RCS_frequency.max()

            print("@{}GHz Min RCS {:0.3f} , Max RCS {:.3f}".format(frequency,RCS_min, RCS_max))

            # image
            xr.plot.imshow(RCS_frequency,cmap="jet",ax=ax,vmin=RCS_frequency.quantile(0.05),vmax=RCS_max,cbar_kwargs={"label":'RCS [dB]'})
            ax.set_title(f'Plot for f[GHz]={frequency}')
        print()

        #
        plt.suptitle(drone_name)
        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Show the plot
        if save_images:
            plt.savefig(os.path.join("..\Drone_RCS_Measurement_Dataset",drone_name + ".pdf"), format="pdf", bbox_inches="tight")
        plt.show(block=False)

    if smooth_images:
        for drone_name, RCS_array in drone_rcs_dictionary.items():
            # Assuming 'f[GHz]' are categories you want to filter on for each subplot

            # Creating 4x2 subplots
            # hardcoded.. should dynamically change
            fig, axs = plt.subplots(3, 5, figsize=(31, 16))  # you can adjust the size as per your requirement

            # Flatten the 4x2 array to easily iterate over it
            axs = axs.flatten()

            unique_frequencies = RCS_array.coords["f[GHz]"].values
            new_az = RCS_array.coords["azimuth"]
            new_el = RCS_array.coords["elevation"]

            new_az= np.linspace(new_az[0], new_az[-1], len(new_az) * 4)
            new_el = np.linspace(new_el[0], new_el[-1], len(new_el) * 4)

            N_az = len(new_az)
            N_el = len(new_el)

            new_az_gr, new_el_gr = np.meshgrid(new_az, new_el)

            new_az_gr = xr.DataArray(new_az_gr.ravel(), dims="points1")
            new_el_gr = xr.DataArray(new_el_gr.ravel(), dims="points1")

            RCS_array_interp = RCS_array.interp(azimuth=new_az_gr,elevation=new_el_gr)
            RCS_array_interp = RCS_array_interp.values.reshape(-1, N_az, N_el, order="f")
            RCS_array_interp = xr.DataArray(RCS_array_interp,coords={"f[GHz]":unique_frequencies,
                                                                     "azimuth":new_az,
                                                                     "elevation":new_el})
            print(f"{drone_name}")
            for i, (frequency, ax) in enumerate(zip(unique_frequencies, axs), 1):
                # Filter data for each unique 'f[GHz]'
                RCS_frequency = RCS_array_interp.loc[frequency,:,:]
                RCS_min, RCS_max = RCS_array_interp.min(), RCS_array_interp.max()

                print("@{}GHz Min RCS {:0.3f} , Max RCS {:.3f}".format(frequency,RCS_min, RCS_max))

                # image
                xr.plot.imshow(RCS_frequency,cmap="jet",ax=ax,vmin=RCS_frequency.quantile(0.05),vmax=RCS_max,cbar_kwargs={"label":'RCS [dB]'})
                ax.set_title(f'Plot for f[GHz]={frequency} Interpolated')

            #
            plt.suptitle(drone_name + " Interpolated")
            # Adjust layout to prevent overlap
            plt.tight_layout()
            # Show the plot
            if save_images:
                plt.savefig(os.path.join("..\Drone_RCS_Measurement_Dataset",drone_name+"_smoothed" + ".pdf"), format="pdf", bbox_inches="tight")
            plt.show(block=False)

def main():
    visualize = False
    DRONE_RCS_FOLDER =  "../Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER,visualize=False,verbose=True)

    if visualize:
        visualize_RCS_dictionary(drone_rcs_dictionary)

    xlim = [-10, 10];  ylim = [-10, 10]; zlim = [50, 150]
    bounding_box = np.array([xlim,ylim,zlim])
    yaw_lim = [-np.pi / 4, np.pi / 4];
    pitch_lim = [-np.pi / 4, np.pi / 4]
    roll_lim = [-np.pi / 4, np.pi / 4]



    num_points = 10000 # Number of points
    num_radars = 3

    # simulate radar positions within some bounding box of limits
    radars = np.column_stack((
        np.random.uniform(xlim[0], xlim[1], num_radars),
        np.random.uniform(ylim[0], ylim[1], num_radars),
        np.zeros((num_radars,))
    ))

    dataset = RCS_TO_DATASET(drone_rcs_dictionary,radars,yaw_lim,pitch_lim,roll_lim,bounding_box,num_points)
    X,y = dataset_to_tensor(dataset,True)

    dataset = RCS_TO_DATASET(drone_rcs_dictionary,radars,yaw_lim,pitch_lim,roll_lim,bounding_box,num_points)

    print("Multi Radar with geometry:",X.shape,y.shape)

    dataset = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,azimuth_center=90,azimuth_spread=5,elevation_center=0,elevation_spread=5,num_points=num_points)

    dataset = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,azimuth_center=90,azimuth_spread=180,elevation_center=0,elevation_spread=190,
                                          num_points=num_points,method="random")


    dataset = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,azimuth_center=90,azimuth_spread=180,elevation_center=0,elevation_spread=190,num_points=num_points,method="random")


    if visualize == True:
        plt.figure()
        RCS_class_0 = dataset["RCS"][dataset["ys"].ravel()==0]
        az_class_0 = dataset["azimuth"][dataset["ys"].ravel()==0]
        el_class_0 = dataset["elevation"][dataset["ys"].ravel()==0]

        plt.scatter(el_class_0,az_class_0,c=RCS_class_0[:,0])
        plt.show()



    dataset_train, dataset_test = dataset_train_test_split(dataset, 0.2)

    X,y = dataset_to_tensor(dataset,True)
    print("Single Point with geometry:",X.shape,y.shape)

    X,y = dataset_to_tensor(dataset,False)
    print("Single Point without geometry:",X.shape,y.shape)


if __name__ == "__main__":
    main()



