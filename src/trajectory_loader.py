from src.geometry import *
from src.data_loader import DRONE_RCS_CSV_TO_XARRAY
from src.noise_generator import add_noise_trajectory,generate_cov
from src.misc import draw_3d_lines_and_points_ref
from time import time
import xarray as xr


import matplotlib.pyplot as plt
from sktime.datatypes import convert
import os,glob,imageio

def simulate_trajectories(init_position,time_step_size,vx,yaw_range,pitch_range,roll_range,TN,N_traj):

    yaws = np.zeros((N_traj,TN))
    pitchs = np.zeros((N_traj,TN))
    rolls = np.zeros((N_traj,TN))
    translations = np.zeros((N_traj,TN,3));
    translations[:,0,:] = init_position

    for i in range(1,TN):
        forward_dir = np.zeros((4,1)); forward_dir[[0,-1],:] = 1;

        yawt = yaw_matrix( yaws[:,i-1])
        pitcht = pitch_matrix(pitchs[:,i-1])
        rollt = roll_matrix(rolls[:,i-1])

        forward_dir = np.matmul(rollt @ pitcht @ yawt, forward_dir).transpose(0, 2, 1)

        translations[:,i,:] = translations[:,i-1,:] + vx * time_step_size * forward_dir[:,0,:-1]
        yaws[:,i] = yaws[:,i-1] + np.random.randn(N_traj)*yaw_range
        rolls[:,i] = rolls[:,i-1] + np.random.randn(N_traj)*roll_range
        pitchs[:,i] = pitchs[:,i-1] + np.random.randn(N_traj)*pitch_range

    return yaws,pitchs,rolls,translations

def simulate_target_trajectory_azim_elev(time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,TN,N_traj):

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws, pitchs, rolls, translations = simulate_trajectories(init_position, time_step_size, vx, yaw_range, pitch_range,
                                                              roll_range, TN, N_traj)
    AZ_ts = np.zeros((N_traj, TN))
    EL_ts = np.zeros((N_traj, TN))

    radars = np.zeros((4, 1));
    radars[-1, :] = 1

    for t in range(TN):
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        relative_distances = np.matmul(inv_yaw @ inv_pitch @ inv_roll @ inv_trans, radars)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        elevation = elevation * 180 / np.pi;
        azimuth = azimuth * 180 / np.pi;

        # how to account for the negative elevations... what does a negative elevation even mean?

        # when we have negative azimuth, it means that it is flipped because symetry... (wrapping effect)
        azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180) % 360

        AZ_ts[:, t] = azimuth.ravel();
        EL_ts[:, t] = elevation.ravel();


    return AZ_ts,EL_ts


def RCS_TO_DATASET_Single_Trajectory(RCS_xarray_dictionary,time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,TN,num_points,verbose=False):
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

        AZ_ts,EL_ts = simulate_target_trajectory_azim_elev(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box, TN,
                                             N_traj=num_points)

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
            valid_sample_idx = ((AZ_ts <= valid_azimuths[1]) & (AZ_ts >= valid_azimuths[0]) & (
                        EL_ts <= valid_elevations[1]) & (EL_ts >= valid_elevations[0]))
            valid_sample_idx = valid_sample_idx.all(-1)
            azimuth_copy = AZ_ts[valid_sample_idx, :]
            elevation_copy = EL_ts[valid_sample_idx, :]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            try:
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))
            except:
                continue

            # number of trajectories x number of time steps x number of frequencies
            RCS_indexed = RCS_indexed.values.T.reshape(-1,TN,N_freqs*N_radars)

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






def simulate_target_trajectory_azim_elev_multi(time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,radars,TN,N_traj):

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws, pitchs, rolls, translations = simulate_trajectories(init_position, time_step_size, vx, yaw_range, pitch_range,
                                                            roll_range, TN, N_traj)
    N_radars = radars.shape[0]
    radars = np.column_stack((radars,np.ones((N_radars, 1))));

    # number of simulate points x number of time steps x number of radars
    AZ_ts = np.zeros((N_traj, TN,N_radars))
    EL_ts = np.zeros((N_traj, TN,N_radars))
    Rho_ts = np.zeros((N_traj, TN,N_radars))

    for t in range(TN):
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        relative_distances = np.matmul(inv_yaw @ inv_pitch @ inv_roll @ inv_trans, radars.T)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        elevation = elevation * 180 / np.pi;
        azimuth = azimuth * 180 / np.pi;

        # how to account for the negative elevations... what does a negative elevation even mean?

        # when we have negative azimuth, it means that it is flipped because symetry... (wrapping effect)
        azimuth[azimuth < 0] = (azimuth[azimuth < 0] + 180) % 360

        AZ_ts[:, t] = azimuth#.ravel();
        EL_ts[:, t] = elevation#.ravel();
        Rho_ts[:,t] = rho

    return AZ_ts,EL_ts,Rho_ts

def RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary,time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,radars,TN,num_points,verbose=False):
    """
    """
    # dictionary to convert label to digit label
    N_radars = radars.shape[0]
    RCSs = [];
    ys = [];
    azimuths = []
    elevations = []
    rhos = []


    if verbose:
        print("CONVERT DRONE RCS DICTIONARY TO X,y DATASET")

    drone_sample_count = {drone_key: num_points for drone_key in RCS_xarray_dictionary.keys()}
    start_time = time()
    while any(np.array(list(drone_sample_count.values())) > 0):

        # get the vector of all drone sample counts
        drone_sample_counts = np.array(list(drone_sample_count.values()))

        # sample the minimum number of points needed across all drones with needed samples still > 0
        num_points = np.min(drone_sample_counts[drone_sample_counts > 0])

        AZ_ts,EL_ts,Rho_ts = simulate_target_trajectory_azim_elev_multi(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box, radars,TN,
                                             N_traj=num_points)

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
            valid_sample_idx = ((AZ_ts <= valid_azimuths[1]) & (AZ_ts >= valid_azimuths[0]) & (
                        EL_ts <= valid_elevations[1]) & (EL_ts >= valid_elevations[0]))
            valid_sample_idx = valid_sample_idx.all(axis=(-2,-1))
            azimuth_copy = AZ_ts[valid_sample_idx,:,:]
            elevation_copy = EL_ts[valid_sample_idx,:,:]
            rho_copy = Rho_ts[valid_sample_idx,:]

            # the RCS_label is the same for all samples from the same Drone RCS_array
            RCS_label = np.array([[drone]])


            # VECTORIZED INDEXING
            try:
                RCS_indexed = RCS_array.interp(azimuth=xr.DataArray(azimuth_copy.ravel(),dims="points"),
                                               elevation=xr.DataArray(elevation_copy.ravel(),dims="points"))
            except:
                continue

            # # number of trajectories x number of time steps x number of frequencies
            # RCS_indexed = RCS_indexed.values.T.reshape(-1,TN,N_freqs*N_radars)
            #
            # number f trajectories x number of time steps x (number of radars * number of frequencies)
            RCS_indexed = RCS_indexed.values.reshape(N_freqs, *azimuth_copy.shape).transpose(1, 2, 3, 0).reshape(-1,TN,N_freqs*N_radars)

            RCSs.append(RCS_indexed)
            ys.append(np.ones((valid_sample_idx.sum(),1))*RCS_label)
            azimuths.append(azimuth_copy)
            elevations.append(elevation_copy)
            rhos.append(rho_copy)

            # update the needed number of samples remaining JUST FOR THE SPECIFIC DRONE
            drone_sample_count[drone] = int(drone_sample_count[drone] - np.sum(valid_sample_idx))

    RCSs = np.vstack(RCSs)
    azimuths = np.vstack(azimuths)
    elevations = np.vstack(elevations)
    rhos = np.vstack(rhos)
    ys = np.vstack(ys)

    dataset = {
        "RCS":RCSs,
                "azimuth":azimuths,
                "elevation":elevations,
                "rho":rhos,
                "ys":ys,
                "n_radars":N_radars,
                "n_freq":N_freqs
            }

    end_time = time()
    if verbose:
        print("Dataset Creation Time: {:.3f}".format(end_time-start_time))

    return dataset

def simulate_target_gif(time_step_size,vx,yaw_range,pitch_range,roll_range,bounding_box,radars,TN):
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    N_traj = 1
    photo_dump = os.path.join("..","results","tmp_photo")
    remove_photo_dump = False
    os.makedirs(photo_dump,exist_ok=True)

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws, pitchs, rolls, translations = simulate_trajectories(init_position, time_step_size, vx, yaw_range, pitch_range,
                                                            roll_range, TN, N_traj)
    N_radars = radars.shape[0]
    radars = np.column_stack((radars,np.ones((N_radars, 1))));

    # number of simulate points x number of time steps x number of radars
    frames = []
    for t in range(TN):
        inv_yaw = inverse_yaw_matrix(yaws[:, t])
        inv_pitch = inverse_pitch_matrix(pitchs[:, t])
        inv_roll = inverse_roll_matrix(rolls[:, t])
        inv_trans = inverse_translation_matrix(translations[:, t, :])

        relative_distances = np.matmul(inv_yaw @ inv_pitch @ inv_roll @ inv_trans, radars.T)

        # number of simulate points  x number of radars x dimension of coordinate (4)
        relative_distances = relative_distances.transpose(0, 2, 1)

        range_, rho, azimuth, elevation = cartesian2spherical(relative_distances[:, :, 0],
                                                              relative_distances[:, :, 1],
                                                              relative_distances[:, :, 2])

        draw_3d_lines_and_points_ref(range_, rho, azimuth, elevation, translations[:,t], yaws[:,t], pitchs[:,t], rolls[:,t], radars[:,:3],
                                     coordinate_system="spherical", ax=ax)

        filename = os.path.join(photo_dump,f"frame_{t}.png")

        # Optionally, you can add grid lines for better visualization
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add some labels (optional)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        plt.savefig(filename)
        frames.append(imageio.imread(filename))
        [line.remove() for line in ax.lines[-N_radars:]]

    plt.close()

    # Save frames as a GIF
    gif_filename = os.path.join("..","results","target_movement.gif")
    imageio.mimsave(gif_filename, frames, duration=0.5)  # Adjust duration as needed
    print(f"GIF saved as '{gif_filename}'")

    if remove_photo_dump:
        for filename in glob.glob(os.path.join(photo_dump,"frame_*")):
            os.remove(filename)



def dataset_to_sktime(dataset):

    dataset["RCS"] =  convert(np.swapaxes(dataset["RCS"],1,2),from_type="numpy3D",to_type="pd-multiindex")
    dataset["elevation"] = convert(np.expand_dims(dataset["elevation"],1),from_type="numpy3D",to_type="pd-multiindex")
    dataset["azimuth"] = convert(np.expand_dims(dataset["azimuth"], 1), from_type="numpy3D",to_type="pd-multiindex")
    dataset["ys"] = dataset["ys"].ravel()

def main():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Qt5Agg')

    TN = 50
    N_traj = 300
    time_step_size = 0.1
    vx = 50
    yaw_range = pitch_range = roll_range = np.pi/15
    xlim = [-50, 50];  ylim = [-50, 50]; zlim = [50, 150]
    bounding_box = np.array([xlim,ylim,zlim])
    num_radars = 10
    SNR = 5

    init_position = np.column_stack((
        np.random.uniform(bounding_box[0,0], bounding_box[0,1], N_traj),
        np.random.uniform(bounding_box[1,0], bounding_box[1,1], N_traj),
        np.random.uniform(bounding_box[2,0], bounding_box[2,1], N_traj)
    ))

    yaws,pitchs,rolls,translations = simulate_trajectories(init_position,time_step_size, vx, yaw_range, pitch_range, roll_range, TN, N_traj)

    # Creating 4x2 subplots
    fig = plt.figure(figsize=(16, 16))
    n_axes = 1

    for i in range(n_axes):
        ax = fig.add_subplot(1,n_axes,i+1, projection = '3d')  # you can adjust the size as per your requirement

        yaw = yaw_matrix(yaws[i])
        pitch = pitch_matrix(pitchs[i])
        roll = roll_matrix(rolls[i])
        trans = translation_matrix(translations[i])

        for j in range(TN):
            plot_target_frames(ax,trans[[j]],roll[[j]],pitch[[j]],yaw[[j]])
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        ax.set_zlabel("Z")
        ax.set_title(f"Traj {i}")
        ax.axis('equal')
    plt.show()

    DRONE_RCS_FOLDER =  "../Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER,visualize=False)

    dataset = RCS_TO_DATASET_Single_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                               time_step_size=time_step_size, vx=vx,
                                               yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,bounding_box=bounding_box,
                                               TN=TN,num_points=N_traj,
                                               verbose=True)


    print(dataset)

    covs_single = generate_cov(TraceConstraint=1, d=15, N=10,
                               blocks=1, color="color",
                               noise_method="random")

    # dataset["RCS"] = add_noise_trajectory(dataset["RCS"], 10, covs_single[0])

    dataset_to_sktime(dataset)

    #### ============== SIMULATE MULTIPLE RADARS ================ #####

    # simulate radar positions within some bounding box of limits
    radars = np.column_stack((
        np.random.uniform(xlim[0], xlim[1], num_radars),
        np.random.uniform(ylim[0], ylim[1], num_radars),
        np.zeros((num_radars,))
    ))

    AZs,ELs,Rhos = simulate_target_trajectory_azim_elev_multi(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box,
                                               radars, TN, N_traj)

    dataset_multi = RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                               time_step_size=time_step_size, vx=vx,
                                               yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,bounding_box=bounding_box,
                                               TN=TN,radars=radars,
                                               num_points=N_traj,
                                               verbose=True)

    print("MULTI")
    add_noise_trajectory(dataset_multi["RCS"],SNR=SNR,cov=covs_single[0],n_radars=num_radars)
    simulate_target_gif(time_step_size, vx, yaw_range, pitch_range, roll_range, bounding_box, radars, TN)

if __name__ == "__main__":
    main()