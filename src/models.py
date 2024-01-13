
import numpy as np



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
        self.use_geometry = use_geometry

        self.n_classes = n_classes
        self.pc = 1/self.n_classes

    def predict_instant(self,clf,dataset,t=0,fusion_method="average"):
        n_radars = dataset["n_radars"]
        n_freq = dataset["n_freq"]

        RCS = dataset["RCS"][:,t,:]
        azimuth = dataset["azimuth"][:,t,:]
        elevation = dataset["elevation"][:,t,:]
        n_classes = len(np.unique(dataset["ys"]))

        predictions = np.zeros((RCS.shape[0],n_classes,n_radars))

        for i in range(n_radars):

            X = RCS[:,(i*n_freq):((i+1)*n_freq)]
            if self.use_geometry:
                azimuth_i = azimuth[:, [i]]
                elevation_i = elevation[:, [i]]
                X = np.hstack((X,azimuth_i,elevation_i))

            y_pred = clf.predict_proba(X)

            predictions[:,:,i] = y_pred

        if fusion_method == "average":
            predictions = predictions.mean(-1)#.argmax(1,keepdims=True)

        elif fusion_method == "max":
            predictions = predictions.max(-1)#.argmax(1,keepdims=True)

        elif fusion_method == "fusion":
            # predictions = np.log(predictions+1e-16).sum(-1)#.argmax(1,keepdims=True)
            num = predictions.prod(-1)
            den = np.sum(predictions.prod(axis=-1),axis=-1,keepdims=True)
            predictions = num/den
        else:
            raise Exception("Not a valid fusion method of multiple radars")

        return predictions

    def predict(self,clf,dataset,fusion_method="fusion",record=True):

        N_traj, N_time, d = dataset["RCS"].shape

        # initialize p_cprev_given_zpast as prior for every trajectory
        p_c_given_past = self.pc #np.tile(self.pc,(dataset["RCS"].shape[0],1))

        p_c_over_time =  None
        if record:
            # Number of Trajectories x Number of Time Steps x Number of Classes
            p_c_over_time = np.zeros((N_traj,N_time,self.n_classes))

        for t in np.arange(N_time):

            # Number of Trajectories x Number of Classes
            p_c_given_z = self.predict_instant(clf,dataset,t=t,fusion_method=fusion_method)

            numerator = p_c_given_z * p_c_given_past

            denominator = np.sum(p_c_given_z * p_c_given_past,axis=-1,keepdims=True)

            p_c_given_all = numerator / denominator

            if record:
                p_c_over_time[:,t,:] = p_c_given_all

            p_c_given_past = p_c_given_all

        return p_c_given_all,p_c_over_time



def main():
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.metrics import classification_report

    from src.noise_generator import add_noise, add_noise_block,add_noise_trajectory
    from src.data_loader import DRONE_RCS_CSV_TO_XARRAY, RCS_TO_DATASET, RCS_TO_DATASET_Single_Point, dataset_to_tensor, \
        dataset_train_test_split

    from src.trajectory_loader import RCS_TO_DATASET_Trajectory
    from src.noise_generator import generate_cov



    from lazypredict.Supervised import LazyClassifier, CLASSIFIERS

    import random

    from sklearn.neural_network import MLPClassifier

    from src.trajectory_loader import target_with_predictions_gif
    from src.misc import radar_grid


    CLASSIFIERS = dict(CLASSIFIERS)

    np.random.seed(12300)
    random.seed(12300)

    classifiers_names = ["XGBClassifier","KNeighborsClassifier","LogisticRegression"]

    classifiers = [CLASSIFIERS[name] for name in classifiers_names]   #+ [MLPClassifier]

    DRONE_RCS_FOLDER = "..\Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encoder = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER)
    drone_names = list(drone_rcs_dictionary.keys())
    n_freq = len(drone_rcs_dictionary[drone_names[0]].coords["f[GHz]"])

    xlim = [-150, 150];  ylim = [-150, 150]; zlim = [200, 300]

    bounding_box = np.array([xlim, ylim, zlim])
    yaw_lim = [-np.pi / 5, np.pi / 5];
    pitch_lim = [-np.pi / 5, np.pi / 5]
    roll_lim = [-np.pi / 5, np.pi / 5]

    n_radars = 4


    radars = radar_grid(n_radars=n_radars,xlim=xlim,ylim=ylim)

    use_geometry = False

    noise_color="color"
    noise_method="random"
    SNR_constraint = 0
    num_points = 10000

    covs_single = generate_cov(TraceConstraint=1, d=n_freq, N=1,
                               blocks=n_radars, color=noise_color,
                               noise_method=noise_method)

    dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                          azimuth_center=90,azimuth_spread=180,
                                          elevation_center=0,elevation_spread=190,
                                          num_points=num_points,method="random",verbose=False)

    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])

    dataset_train, dataset_test = dataset_train_test_split(dataset_single)



    X_train, y_train = dataset_to_tensor(dataset_train, use_geometry)
    X_test, y_test = dataset_to_tensor(dataset_test, use_geometry)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, classifiers=classifiers)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    print(models.to_string())

    dataset_multi = RCS_TO_DATASET(drone_rcs_dictionary, radars, yaw_lim, pitch_lim, roll_lim, bounding_box,
                                   num_points=100)

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

    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])





    dataset_train, dataset_test = dataset_train_test_split(dataset_single)

    #================= TEST DISTRIBUTED RADAR CLASSIFIER =================#
    TN = 100
    N_traj = 1000
    time_step_size = 0.1
    vx = 50
    yaw_range , pitch_range , roll_range = np.pi/8,np.pi/15,0
    # xlim = [-50, 50];  ylim = [-50, 50]; zlim = [150, 300]
    xlim = [-150, 150];  ylim = [-150, 150]; zlim = [200, 300]

    bounding_box = np.array([xlim,ylim,zlim])
    plotting_args = {"arrow_length": 10, "arrow_linewidth": 2}

    dataset_multi = RCS_TO_DATASET_Trajectory(RCS_xarray_dictionary=drone_rcs_dictionary,
                                              time_step_size=time_step_size, vx=vx,
                                              yaw_range=yaw_range, pitch_range=pitch_range, roll_range=roll_range,
                                              bounding_box=bounding_box,
                                              TN=TN, radars=radars,
                                              num_points=N_traj,
                                              verbose=True)

    dataset_multi["RCS"] = add_noise_trajectory(dataset_multi["RCS"],SNR_constraint,covs_single[0],n_radars)


    # ============== GENERATE FIGURES ============= #
    drc =  distributed_recursive_classifier(len(label_encoder.classes_),use_geometry=use_geometry)

    _,pred_logistic_history = drc.predict(clf.models["LogisticRegression"],dataset_multi)
    _,pred_xgb_history = drc.predict(clf.models["XGBClassifier"],dataset_multi)

    dataset_multi["n_radars"] = 1
    _,pred_logistic_history_1 = drc.predict(clf.models["LogisticRegression"],dataset_multi)
    _,pred_xgb_history_1 = drc.predict(clf.models["XGBClassifier"],dataset_multi)

    import matplotlib.pyplot as plt
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
    plt.show()


    # =============== visualize fusion differences ======================= #
    dataset_multi["n_radars"] = radars.shape[0]
    drc =  distributed_recursive_classifier(len(label_encoder.classes_),use_geometry=use_geometry)
    _,pred_xgb_history_fuse = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="fusion")
    _,pred_xgb_history_avg = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="average")
    _,pred_xgb_history_max = drc.predict(clf.models["XGBClassifier"],dataset_multi,fusion_method="max")


    plt.figure()
    xgb_history_fuse = pred_xgb_history_fuse.argmax(-1)
    xgb_accuracy_fuse = (xgb_history_fuse == dataset_multi["ys"]).mean(0)

    xgb_history_avg = pred_xgb_history_avg.argmax(-1)
    xgb_accuracy_avg = (xgb_history_avg == dataset_multi["ys"]).mean(0)

    xgb_history_max= pred_xgb_history_max.argmax(-1)
    xgb_accuracy_max= (xgb_history_max == dataset_multi["ys"]).mean(0)

    plt.plot(xgb_accuracy_fuse,color="purple",marker="o",linestyle="-")
    plt.plot(xgb_accuracy_avg,color="purple",marker="+",linestyle="-")
    plt.plot(xgb_accuracy_max,color="purple",marker="x",linestyle="-")



    plt.ylabel("Accuracy");
    plt.xlabel("Time Step");
    plt.legend([f"XGBoost {n_radars} Fuse",
                f"XGBoost {n_radars} Avg",
                f"XGBoost {n_radars} Max"])

    plt.title(f"RBC Fusion - SNR={SNR_constraint}")
    plt.show()


    target_with_predictions_gif(dataset_multi,pred_xgb_history_fuse,radars,plotting_args=plotting_args)


if __name__ == "__main__":
    main()




