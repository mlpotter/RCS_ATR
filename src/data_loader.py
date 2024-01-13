import numpy as np
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

# TODO: WRITE CODE TO SAVE, LOAD PICKLE FILES
# TODO: note that parrot_HH and MAVIC_HH have elevation from 0-180... is this mistake?
def DRONE_RCS_CSV_TO_XARRAY(path,visualize=False,verbose=False):
    if verbose:
        print("LOAD RCS DATA FROM CSV FILES")
    # get all the .csv files with the drones RCS values
    csv_files = glob.glob(os.path.join(path,"*.csv"))

    # initialize drone dictionary {"drone name": xarray stacked over frequency}
    drone_dictionary = {}

    # iterate through each drone RCS csv file
    for file_path in csv_files:
        data = pd.read_csv(file_path)

        ## make sure consistent organization amongst data csv files...
        data = data.sort_values(by=["f[GHz]", "phi[deg]", "theta[deg]"], ascending=True).reset_index(drop=True)

        drone_name = file_path.split(os.sep)[-1].replace(".csv", "")

        if visualize:
            data["RCS[dB]"].hist(by=data["f[GHz]"], figsize=(30, 30))
            plt.suptitle(drone_name)
            plt.show()

        if drone_name in ["Mavic_HH","Parrot_HH","battery_HH"]:
            continue

        if "HH" not in drone_name:
            continue


        if verbose:
            descriptive_df = pd.concat((pd.DataFrame(data.min(0)).T, pd.DataFrame(data.max(0)).T))
            descriptive_df.index = ["min", "max"]
            print(f"Loaded {drone_name}")
            print(descriptive_df.to_string())

        # Assuming 'f[GHz]' are categories you want to filter on for each subplot
        unique_frequencies = data['f[GHz]'].unique()
        # print("Unique Frequencies: ",unique_frequencies)
        # list of RCS xarray files, where RCS xarray are for each frequency
        RCS_arrays = []
        for i, frequency in enumerate(unique_frequencies, 1):
            specific_data = data[data['f[GHz]'] == frequency]

            # Extracting specific data columns for plotting
            # theta is elevation, phi is azimuth
            theta = specific_data['theta[deg]']
            phi = specific_data['phi[deg]']
            rcs = specific_data['RCS[dB]']  # color coding based on RCS values

            # print("@{}GHz Min RCS {:0.3f} , Max RCS {:.3f}".format(frequency,rcs.min(), rcs.max()))

            # convert dataframe to xarray, where x coordinates are azimuth and y coordinates are "elevation"
            coords = {"azimuth": phi.unique(), "elevation": theta.unique()}

            n_azimuth = len(phi.unique())
            n_elevation = len(theta.unique())

            # vectorized version of for loop above, but need to sanity check at some point that the RESHAPE does not mess up the coordination between azimuth, elevataion, and rcs
            RCS_array = xr.DataArray(rcs.values.reshape(n_azimuth,n_elevation), coords=coords, dims=["azimuth", "elevation"])

            RCS_array.name = f"{frequency}[GHz]"
            RCS_arrays.append(RCS_array.expand_dims("f[GHz]"))


        drone_dictionary[drone_name] = xr.concat(RCS_arrays, pd.Index(unique_frequencies,name="f[GHz]"))

        if verbose:
            print("RCS shape: ",drone_dictionary[drone_name].shape)


    label_dictionary = LabelEncoder().fit(list(drone_dictionary.keys()))
    print(list(zip(label_dictionary.transform(list(drone_dictionary.keys())),list(drone_dictionary.keys()))))
    drone_dictionary = {label_dictionary.transform([k])[0]:v for k,v in drone_dictionary.items()}

    return drone_dictionary,label_dictionary

def simulate_target_azim_elev(radars,yaw_lim,pitch_lim,roll_lim,bounding_box,num_points):
    yaws = np.random.uniform(yaw_lim[0], yaw_lim[1], num_points)
    pitchs = np.random.uniform(pitch_lim[0], pitch_lim[1], num_points)
    rolls = np.random.uniform(roll_lim[0], roll_lim[1], num_points)

    translations = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], num_points),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], num_points),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], num_points)
    ))

    range_, rho, azimuth, elevation = calculate_3d_angles_ref(translations, yaws, pitchs, rolls, radars)

    elevation = elevation * 180/np.pi; azimuth = azimuth * 180/np.pi;

    # how to account for the negative elevations... what does a negative elevation even mean?

    # when we have negative azimuth, it means that it is flipped because symetry... (wrapping effect)
    azimuth[azimuth < 0]  = (azimuth[azimuth < 0] + 180) % 360

    return azimuth,elevation

# TODO: Incorperate Dataset metadata to make dictionary data object ... such as the radars positions, yaws, bounding box, etc
def RCS_TO_DATASET(RCS_xarray_dictionary,radars,yaw_lim,pitch_lim,roll_lim,bounding_box,num_points,verbose=False):
    """
    """

    N_radars = radars.shape[0]
    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []

    drone_sample_count = {drone_key:num_points for drone_key in RCS_xarray_dictionary.keys()}
    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO X,y DATASET")
    start_time = time()
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts>0])

        # generate the azimuth and elevations of the target with respect to radar line of sight.
        # number of points x number of radars
        azimuth, elevation = simulate_target_azim_elev(radars, yaw_lim, pitch_lim, roll_lim, bounding_box,num_points)

        # iterate through each drone's RCS "stack"
        for drone,RCS_array in RCS_xarray_dictionary.items():
            # print(drone)

            # check if this drone has enough samples already...
            if drone_sample_count[drone] <= 0:
                continue

            # all the valid azimuth and elevationns for particular RCS_array
            azimuth_axes = RCS_array.coords["azimuth"].values
            elevation_axes = RCS_array.coords["elevation"].values
            frequency_axes = RCS_array.coords["f[GHz]"].values

            N_freqs = len(frequency_axes)

            # boundary of azimuth and elevations to check for valid sample generated
            valid_azimuths, valid_elevations = [np.min(azimuth_axes),np.max(azimuth_axes)],[np.min(elevation_axes),np.max(elevation_axes)]

            # the random azimuth and elevation MUST BE WITHIN RANGE OF THE REAL DATA!
            # a sample is only valid if ALL the azimuths and elevations wrt each radar are VALID
            valid_sample_idx = ((azimuth <= valid_azimuths[1]) & (azimuth >= valid_azimuths[0]) & (elevation <= valid_elevations[1]) & (elevation >= valid_elevations[0]))
            valid_sample_idx = valid_sample_idx.all(-1)
            azimuth_copy = azimuth[valid_sample_idx,:]
            elevation_copy = elevation[valid_sample_idx,:]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            # number of frequencies x (number of samples * number of radars)
            try:
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))
            except:
                continue

            # number of samples x (number of radars * number of frequencies)
            RCS_indexed = RCS_indexed.values.T.reshape(-1,N_freqs*N_radars)

            RCSs.append(RCS_indexed)
            ys.append(np.ones((valid_sample_idx.sum(),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)

            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - np.sum(valid_sample_idx))

        if verbose:
            print(drone_sample_count)
    RCSs = np.vstack(RCSs)

    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)
    ys = np.vstack(ys)

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
def RCS_TO_DATASET_Single_Point(RCS_xarray_dictionary,azimuth_center,azimuth_spread,elevation_center,elevation_spread,num_points,method="target",verbose=False):
    """
    """
    # dictionary to convert label to digit label
    N_radars = 1

    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []

    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO X,y DATASET")

    drone_sample_count = {drone_key: num_points for drone_key in RCS_xarray_dictionary.keys()}
    start_time = time()
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts > 0])


        if method == "target":
            # generate the azimuth and elevations of the target with respect to radar line of sight.
            # number of points x number of radars
            azimuth = azimuth_spread * np.random.randn(num_points, 1) + azimuth_center

            # since we only have azimuth from 0-180, make sure to account for symetry of drone
            azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180) % 360

            elevation = elevation_spread * np.random.randn(num_points, 1) + elevation_center

        elif method == "random":
            half_spread_az = azimuth_spread/2
            half_spread_el = elevation_spread/2
            azimuth = np.random.uniform(azimuth_center-half_spread_az,azimuth_center+half_spread_az,size=(num_points,1))
            elevation = np.random.uniform(elevation_center-half_spread_el,elevation_center+half_spread_el,size=(num_points,1))

            # since we only have azimuth from 0-180, make sure to account for symetry of drone
            azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180) % 360

        else:
            raise Exception("Sorry, this method is not valid")

        # iterate through each drone's RCS "stack"
        for drone, RCS_array in RCS_xarray_dictionary.items():
            # print(drone)

            # check if this drone has enough samples already...
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
            valid_sample_idx = ((azimuth <= valid_azimuths[1]) & (azimuth >= valid_azimuths[0]) & (
                        elevation <= valid_elevations[1]) & (elevation >= valid_elevations[0]))
            valid_sample_idx = valid_sample_idx.all(-1)
            azimuth_copy = azimuth[valid_sample_idx, :]
            elevation_copy = elevation[valid_sample_idx, :]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            try:
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))
            except:
                continue

            RCS_indexed = RCS_indexed.values.T.reshape(-1,N_freqs*N_radars)

            RCSs.append(RCS_indexed)
            ys.append(np.ones((valid_sample_idx.sum(),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)

            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - np.sum(valid_sample_idx))

    RCSs = np.vstack(RCSs)

    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)
    ys = np.vstack(ys)

    dataset = {
        "RCS":RCSs,
                "azimuth":azimuths,
                "elevation":elevations,
                "ys":ys,
                "n_radars":1,
                "n_freq":N_freqs
            }

    end_time = time()
    if verbose:
        print("Dataset Creation Time: {:.3f}".format(end_time-start_time))

    return dataset

def dataset_to_tensor(dataset,use_geometry):
    RCS = dataset["RCS"]

    X = RCS

    if use_geometry:
        azimuth = dataset["azimuth"]
        elevation = dataset["elevation"]
        X = np.hstack((X,azimuth,elevation))

    y = dataset["ys"]
    return X,y

def dataset_train_test_split(dataset,test_size=0.2):
    n_radars = dataset["n_radars"]
    n_freq = dataset["n_freq"]

    RCS = dataset["RCS"]
    azimuth = dataset["azimuth"]
    elevation = dataset["elevation"]
    y = dataset["ys"]


    RCS_train, RCS_test, Az_train,Az_test,El_train,El_test, y_train, y_test = train_test_split(RCS,azimuth,elevation, y, test_size=test_size, random_state=123, stratify=y.ravel())

    dataset_train = {
                "RCS":RCS_train,
                "azimuth":Az_train,
                "elevation":El_train,
                "ys":y_train,
                "n_radars":n_radars,
                "n_freq":n_freq
            }

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
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER,visualize=False)

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



