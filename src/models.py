
import numpy as np
from scipy import stats as ss


class distributed_classifier(object):
    def __init__(self,clf,use_geometry=False):
        self.clf = clf
        self.use_geometry = use_geometry
    def predict(self,dataset,fusion_method="average"):
        n_radars = dataset["n_radars"]
        n_freq = dataset["n_freq"]

        RCS = dataset["RCS"]
        azimuth = dataset["azimuth"]
        elevation = dataset["elevation"]
        n_classes = len(np.unique(dataset["ys"]))

        predictions = np.zeros((RCS.shape[0],n_classes,n_radars))

        for i in range(n_radars):

            X = RCS[:,(i*n_freq):((i+1)*n_freq)]
            if self.use_geometry:
                azimuth_i = azimuth[:, [i]]
                elevation_i = elevation[:, [i]]
                X = np.hstack((X,azimuth_i,elevation_i))

            y_pred = self.clf.predict_proba(X)

            predictions[:,:,i] = y_pred

        if fusion_method == "average":
            predictions = predictions.mean(-1).argmax(1,keepdims=True)

        elif fusion_method == "max":
            predictions = predictions.max(-1).argmax(1,keepdims=True)

        elif fusion_method == "fusion":
            predictions = np.log(predictions+1e-16).sum(-1).argmax(1,keepdims=True)
        else:
            raise Exception("Not a valid fusion method of multiple radars")

        return predictions

class distributed_recursive_classifier(object):
    def __init__(self,n_classes,use_geometry=False):
        """
        @param n_classes: int or float. The number of target class labels
        @param use_geometry: bool. True - include the azimuth and elevation and covariates. False - only use RCS as features
        """

        self.use_geometry = use_geometry

        # the number of target class labels
        self.n_classes = n_classes

        # assume marginal distribution p(c) and prior p(c) is 1/|C|
        self.pc = 1/self.n_classes

    def predict_instant(self,clf,dataset,t=0,fusion_method="average"):
        """
        @param clf: the ML model as a scikit-learn object
        @param dataset: dictionary {"RCS": RCSs array of shape Number Trajectories x Number of time steps x (Number of radars * Number of frequencies)
        @param t: the time step/slice to predict on
        @param fusion_method: str. Average - the softmax of classification vectors from each radar. Max - the maximum across all classification vectors from each radar. Fusion - the bayesian fusion method.
        @return: numpy array of probability predicions. Number of trajectories x Number of Classes
        """

        # the number of radars in the experiment
        n_radars = dataset["n_radars"]

        # the number of unique frequencies in the data
        n_freq = dataset["n_freq"]

        # numpy array of RCS data. Number of Trajectories x (Number of radars * Number of Frequencies)
        RCS = dataset["RCS"][:,t,:]

        # numpy array of the azimuth and eleevation data. Number of Trajectories x Number of radars
        azimuth = dataset["azimuth"][:,t,:]
        elevation = dataset["elevation"][:,t,:]

        # number of target class labels
        n_classes = len(np.unique(dataset["ys"]))

        # initialize empty array to hold all predictions across samples and radars
        # Number of trajectories x Number of Classes x Number of Radars
        predictions = np.zeros((RCS.shape[0],n_classes,n_radars))

        # Iterate through each radar to make a individual prediction
        for i in range(n_radars):

            # subset the RCS values corresponding to the one radar.
            # Number of trajectories x Number of Frequencies
            X = RCS[:,(i*n_freq):((i+1)*n_freq)]

            # if True, append the azimuth and elevation to the input feature X
            # Number of trajectories x (Number of Frequencies + 2)
            if self.use_geometry:
                azimuth_i = azimuth[:, [i]]
                elevation_i = elevation[:, [i]]
                X = np.hstack((X,azimuth_i,elevation_i))

            # Have the individual ML model make a prediction
            # number of trajectories x number of classes
            y_pred = clf.predict_proba(X)

            # save the radar prediction for all the trajectories at time step t
            # Number of trajectories x Numbber of Classes
            predictions[:,:,i] = y_pred

        # basec on the fusion method we take a soft-vote, maximum, or bayesian fusion
        if fusion_method == "average":
            predictions = predictions.mean(-1)#.argmax(1,keepdims=True)

        elif fusion_method == "max":
            predictions = predictions.max(-1)#.argmax(1,keepdims=True)

        elif fusion_method == "fusion":
            # predictions = np.log(predictions+1e-16).sum(-1)#.argmax(1,keepdims=True)
            num = predictions.prod(axis=-1)
            den = np.sum(num,axis=-1,keepdims=True)
            predictions = num/den

        elif fusion_method == "hardvote":
            modes = ss.mode(predictions.argmax(1), axis=1).mode
            predictions =  np.ones(predictions.shape[:-1]) * 1e-4 * self.n_classes
            predictions[np.arange(predictions.shape[0]), modes] = 1.0 - (self.n_classes-1)/(self.n_classes) * 1e-4

        elif fusion_method== "random":
            predictions = np.random.dirichlet(alpha=np.ones(self.n_classes),size=(predictions.shape[0],))

        else:
            raise Exception("Not a valid fusion method of multiple radars")

        # Number of Trajectories x Number of Classes
        return predictions

    def predict(self,clf,dataset,fusion_method="fusion",record=True):
        """

        @param clf: the ML model as a scikit-learn object
        @param dataset: dictionary {"RCS": RCSs array of shape Number Trajectories x Number of time steps x (Number of radars * Number of frequencies)
        @param fusion_method: str. Average - the softmax of classification vectors from each radar. Max - the maximum across all classification vectors from each radar. Fusion - the bayesian fusion method.
        @param record: bool. Whether to keep all the fused probabilities over the entire trajectory. Number of Trajectories x Number of Time Steps x Number of classes
        @return:
        """
        # the needed dimensions
        N_traj, N_time, d = dataset["RCS"].shape

        # initialize p_cprev_given_zpast as prior for every trajectory
        p_c_given_past = self.pc #np.tile(self.pc,(dataset["RCS"].shape[0],1))

        p_c_over_time =  None
        if record:
            # Number of Trajectories x Number of Time Steps x Number of Classes
            p_c_over_time = np.zeros((N_traj,N_time,self.n_classes))

        # iterate through every time step of the trajectory
        for t in np.arange(N_time):

            # Number of Trajectories x Number of Classes
            # get the instant bayesian fused discrimination classification p(c|z_t)
            p_c_given_z = self.predict_instant(clf,dataset,t=t,fusion_method=fusion_method)

            # Number of Trajectories x Number of Classes
            # get the numerator p(c|zt)p(c|z1,...,zt-1)
            numerator = p_c_given_z * p_c_given_past

            # get the denominator sum(p(c|zt)p(c|z1,...,zt-1)) over c
            # Number of Trajectories x 1
            denominator = np.sum(numerator,axis=-1,keepdims=True) #np.sum(p_c_given_z * p_c_given_past,axis=-1,keepdims=True)

            # Number of Trajectories x Number of Classes
            # The posterior over all time steps and features p(C|z1,...zt)
            p_c_given_all = numerator / denominator

            if record:
                p_c_over_time[:,t,:] = p_c_given_all

            # the current posterior becomes the next prior
            p_c_given_past = p_c_given_all

        return p_c_given_all,p_c_over_time



def main():
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.metrics import classification_report

    from src.noise_generator import add_noise, add_noise_block,add_noise_trajectory,add_rice_noise
    from src.data_loader import DRONE_RCS_CSV_TO_XARRAY, RCS_TO_DATASET, RCS_TO_DATASET_Single_Point, dataset_to_tensor, \
        dataset_train_test_split

    from src.trajectory_loader import RCS_TO_DATASET_Trajectory
    from src.noise_generator import generate_cov



    from lazypredict.Supervised import LazyClassifier, CLASSIFIERS

    import random
    import xarray as xr
    from sklearn.neural_network import MLPClassifier

    from src.trajectory_loader import target_with_predictions_gif
    from src.misc import radar_grid
    import matplotlib.pyplot as plt

    exponentiate = True
    n_radars = 4
    use_geometry = False
    make_gif = True
    K=10

    noise_color="color"
    noise_method="random"
    SNR_constraint = 0
    num_points = 10000

    CLASSIFIERS = dict(CLASSIFIERS)

    np.random.seed(1)
    random.seed(1)

    classifiers_names = ["XGBClassifier","KNeighborsClassifier","LogisticRegression"]

    classifiers = [CLASSIFIERS[name] for name in classifiers_names]   #+ [MLPClassifier]

    DRONE_RCS_FOLDER = "..\Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER,exponentiate=exponentiate)
    drone_names = list(drone_rcs_dictionary.keys())
    n_freq = len(drone_rcs_dictionary[drone_names[0]].coords["f[GHz]"])

    xlim = [-150, 150];  ylim = [-150, 150]; zlim = [200, 300]

    bounding_box = np.array([xlim, ylim, zlim])
    yaw_lim = [-np.pi / 5, np.pi / 5];
    pitch_lim = [-np.pi / 5, np.pi / 5]
    roll_lim = [-np.pi / 5, np.pi / 5]



    radars = radar_grid(n_radars=n_radars,xlim=xlim,ylim=ylim)


    covs_single = generate_cov(TraceConstraint=1, d=n_freq, N=1,
                               blocks=n_radars, color=noise_color,
                               noise_method=noise_method)

    dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                          azimuth_center=90,azimuth_spread=180,
                                          elevation_center=0,elevation_spread=190,
                                          num_points=num_points,method="random",verbose=False)

    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])
    # dataset_single["RCS"] = add_rice_noise(dataset_single["RCS"],SNR=SNR_constraint,K=K)

    dataset_train, dataset_test = dataset_train_test_split(dataset_single)



    X_train, y_train = dataset_to_tensor(dataset_train, use_geometry)
    X_test, y_test = dataset_to_tensor(dataset_test, use_geometry)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, classifiers=classifiers)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    print(models.to_string())
    #
    dataset_multi = RCS_TO_DATASET(drone_rcs_dictionary, radars, yaw_lim, pitch_lim, roll_lim, bounding_box,
                                   num_points=100)

    # dataset_multi["RCS"] = add_rice_noise(dataset_multi["RCS"],SNR=SNR_constraint,K=K)
    #
    #
    dataset_multi["RCS"] = add_noise_block(dataset_multi["RCS"],SNR_constraint, covs_single[0],
                                           n_radars)


    for classifiers_name in classifiers_names:

        distributed_clf = distributed_classifier(clf.models[classifiers_name],use_geometry)

        print(f"Distributed {classifiers_name} CLASSIFICATION")
        y_pred_avg = distributed_clf.predict(dataset_multi,fusion_method="average")
        print(classification_report(dataset_multi["ys"],y_pred_avg))





    dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                          azimuth_center=90,azimuth_spread=180,
                                          elevation_center=0,elevation_spread=190,
                                          num_points=num_points,method="random")
    # dataset_single["RCS"] = add_rice_noise(dataset_single["RCS"],SNR=SNR_constraint,K=K)
    #
    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])





    dataset_train, dataset_test = dataset_train_test_split(dataset_single)

    #================= TEST DISTRIBUTED RADAR CLASSIFIER =================#
    TN = 100
    N_traj = 100
    time_step_size = 0.1
    vx = 50
    yaw_range , pitch_range , roll_range = np.pi/15,np.pi/20,0
    # xlim = [-50, 50];  ylim = [-50, 50]; zlim = [150, 300]
    xlim = [-150, 150];  ylim = [-150, 150]; zlim = [200, 300]
    #
    bounding_box = np.array([xlim,ylim,zlim])
    plotting_args = {"arrow_length": 15, "arrow_linewidth": 2}

    dataset_multi = RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                              time_step_size=time_step_size, vx=vx,
                                              yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,
                                              bounding_box=bounding_box,
                                              TN=TN, radars=radars,
                                              num_points=N_traj,
                                              verbose=True)


    # plot a mapping of azimuth and elevation to RCS
    sample_idx = 0
    fig,axes = plt.subplots(1,3,figsize=(15,5))
    axes[0].plot(dataset_multi["RCS"][sample_idx,:,0].ravel(),'b-')

    axes[0].plot(drone_rcs_dictionary[dataset_multi["ys"][sample_idx].item()].interp(azimuth=xr.DataArray(dataset_multi["azimuth"][sample_idx,:,0],dims="points1"),
                                                                                     elevation=xr.DataArray(dataset_multi["elevation"][sample_idx,:,0],dims="points1")).loc[26].values,',r--')
    axes[0].set_title("RCS")
    axes[1].plot(dataset_multi["azimuth"][sample_idx,:,0].ravel())
    axes[1].plot(dataset_multi["elevation"][sample_idx,:,0].ravel())
    axes[1].legend(["Azimuth","Elevation"])
    axes[1].set_title("Target Orientation")

    temp_data = 10 * np.log10(drone_rcs_dictionary[dataset_multi["ys"][sample_idx].item()].loc[26])
    vmax = np.max(temp_data)*1.5
    vmin = np.min(temp_data)*0.5
    xr.plot.imshow(temp_data,vmin=vmin,vmax=vmax,ax=axes[2],cmap="jet")
    axes[2].plot(dataset_multi["elevation"][sample_idx,:,0].ravel(),dataset_multi["azimuth"][sample_idx,:,0].ravel(),linewidth=5,color="k")
    axes[2].plot(dataset_multi["elevation"][sample_idx,0,0].ravel(),dataset_multi["azimuth"][sample_idx,0,0].ravel(),markersize=20,color="purple",marker="*")
    axes[2].plot(dataset_multi["elevation"][sample_idx,-1,0].ravel(),dataset_multi["azimuth"][sample_idx,-1,0].ravel(),markersize=20,color="purple",marker="o")
    plt.show()

    # dataset_multi["RCS"] = add_rice_noise(dataset_multi["RCS"],SNR=SNR_constraint,K=K)
    dataset_multi["RCS"] = add_noise_trajectory(dataset_multi["RCS"],SNR_constraint,covs_single[0])


    # ============== GENERATE FIGURES ============= #
    drc =  distributed_recursive_classifier(len(label_encoder.classes_),use_geometry=use_geometry)

    _,pred_logistic_history = drc.predict(clf.models["LogisticRegression"],dataset_multi,fusion_method="fusion")
    _,pred_xgb_history = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="fusion")

    dataset_multi["n_radars"] = 1
    _,pred_logistic_history_1 = drc.predict(clf.models["LogisticRegression"],dataset_multi,fusion_method="fusion")
    _,pred_xgb_history_1 = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="fusion")

    plt.figure()
    xgb_history = pred_xgb_history.argmax(-1)
    xgb_accuracy = (xgb_history == dataset_multi["ys"]).mean(0)

    logistic_history = pred_logistic_history.argmax(-1)
    logistic_accuracy = (logistic_history == dataset_multi["ys"]).mean(0)

    logistic_history_1 = pred_logistic_history_1.argmax(-1)
    logistic_accuracy_1 = (logistic_history_1 == dataset_multi["ys"]).mean(0)

    xgb_history_1 = pred_xgb_history_1.argmax(-1)
    xgb_accuracy_1 = (xgb_history_1 == dataset_multi["ys"]).mean(0)

    plt.plot(xgb_accuracy,color="purple",marker="o",linestyle="-")
    plt.plot(logistic_accuracy,'bx-')
    plt.plot(xgb_accuracy_1,color="purple",marker="+",linestyle="-")
    plt.plot(logistic_accuracy_1,'b^-')


    plt.ylabel("Accuracy");
    plt.xlabel("Time Step");
    plt.legend([f"XGBoost {n_radars}",
                f"Logistic Regression {n_radars}",
                f"XGBoost {1}",
                f"Logistic Regression {1}"])

    plt.title(f"RBC Models - SNR={SNR_constraint}")
    plt.ylim([0,1.05])

    plt.show()


    # =============== visualize fusion differences ======================= #
    dataset_multi["n_radars"] = radars.shape[0]
    drc =  distributed_recursive_classifier(len(label_encoder.classes_),use_geometry=use_geometry)
    _,pred_xgb_history_fuse = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="fusion")
    _,pred_xgb_history_avg = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="average")
    _,pred_xgb_history_max = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="max")
    _,pred_xgb_history_hardvote = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="hardvote")
    _,pred_xgb_history_random = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="random")


    plt.figure()
    xgb_history_fuse = pred_xgb_history_fuse.argmax(-1)
    xgb_accuracy_fuse = (xgb_history_fuse == dataset_multi["ys"]).mean(0)

    xgb_history_avg = pred_xgb_history_avg.argmax(-1)
    xgb_accuracy_avg = (xgb_history_avg == dataset_multi["ys"]).mean(0)

    xgb_history_max= pred_xgb_history_max.argmax(-1)
    xgb_accuracy_max= (xgb_history_max == dataset_multi["ys"]).mean(0)


    xgb_history_hardvote= pred_xgb_history_hardvote.argmax(-1)
    xgb_accuracy_hardvote= (xgb_history_hardvote == dataset_multi["ys"]).mean(0)

    xgb_history_random =  pred_xgb_history_random.argmax(-1)
    xgb_accuracy_random= (xgb_history_random == dataset_multi["ys"]).mean(0)

    plt.plot(xgb_accuracy_fuse,color="purple",marker="o",linestyle="-",label=f"XGBoost {n_radars} Fuse")
    plt.plot(xgb_accuracy_avg,color="purple",marker="+",linestyle="-",label=f"XGBoost {n_radars} Avg")
    plt.plot(xgb_accuracy_max,color="purple",marker="x",linestyle="-",label=f"XGBoost {n_radars} Max")
    plt.plot(xgb_accuracy_hardvote,color="purple",marker="*",linestyle="-",label=f"XGBoost {n_radars} Hard")
    plt.plot(xgb_accuracy_random,color="purple",marker="^",linestyle="-",label=f"XGBoost {n_radars} Random")

    plt.ylim([0,1.05])



    plt.ylabel("Accuracy");
    plt.xlabel("Time Step");
    plt.legend()

    plt.title(f"RBC Fusion - SNR={SNR_constraint}")
    plt.show()

    if make_gif:
        target_with_predictions_gif(dataset_multi,pred_xgb_history_fuse,radars,plotting_args=plotting_args)


if __name__ == "__main__":
    main()




