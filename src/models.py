
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
            pass

        return predictions

def main():
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.metrics import classification_report

    from src.noise_generator import add_noise, add_noise_block
    from src.data_loader import DRONE_RCS_CSV_TO_XARRAY, RCS_TO_DATASET, RCS_TO_DATASET_Single_Point, dataset_to_tensor, \
        dataset_train_test_split
    from src.noise_generator import generate_cov

    from lazypredict.Supervised import LazyClassifier, CLASSIFIERS

    import random


    CLASSIFIERS = dict(CLASSIFIERS)

    np.random.seed(123)
    random.seed(123)

    classifiers_names = ["XGBClassifier","KNeighborsClassifier","LogisticRegression"]

    classifiers = [CLASSIFIERS[name] for name in classifiers_names]  # + [MLPClassifier]

    DRONE_RCS_FOLDER = "..\Drone_RCS_Measurement_Dataset"
    drone_rcs_dictionary,label_encodery = DRONE_RCS_CSV_TO_XARRAY(DRONE_RCS_FOLDER)
    drone_names = list(drone_rcs_dictionary.keys())
    n_freq = len(drone_rcs_dictionary[drone_names[0]].coords["f[GHz]"])

    xlim = [0, 20];
    ylim = [0, 20];
    zlim = [50, 200];
    bounding_box = np.array([xlim, ylim, zlim])
    yaw_lim = [-np.pi / 5, np.pi / 5];
    pitch_lim = [-np.pi / 5, np.pi / 5]
    roll_lim = [-np.pi / 5, np.pi / 5]

    n_radars = 20

    radars = np.column_stack((
        np.random.uniform(xlim[0], xlim[1],n_radars),
        np.random.uniform(ylim[0], ylim[1], n_radars),
        np.zeros((n_radars,))
    ))

    use_geometry = False

    noise_color="color"
    noise_method="random"
    SNR_constraint = 1000

    covs_single = generate_cov(TraceConstraint=1, d=n_freq, N=1,
                               blocks=n_radars, color=noise_color,
                               noise_method=noise_method)

    dataset_single = RCS_TO_DATASET_Single_Point(drone_rcs_dictionary,
                                          azimuth_center=90,azimuth_spread=180,
                                          elevation_center=0,elevation_spread=190,
                                          num_points=10000,method="random")

    dataset_single["RCS"] = add_noise(dataset_single["RCS"],SNR_constraint,covs_single[0])

    dataset_train, dataset_test = dataset_train_test_split(dataset_single)



    X_train, y_train = dataset_to_tensor(dataset_train, use_geometry)
    X_test, y_test = dataset_to_tensor(dataset_test, use_geometry)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, classifiers=classifiers)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    print(models.to_string())

    dataset_multi = RCS_TO_DATASET(drone_rcs_dictionary, radars, yaw_lim, pitch_lim, roll_lim, bounding_box,
                                   num_points=100)

    dataset_multi["RCS"] = add_noise_block(dataset_multi["RCS"], SNR_constraint, covs_single[0],
                                           n_radars)


    for classifiers_name in classifiers_names:

        distributed_clf = distributed_classifier(clf.models[classifiers_name],use_geometry)

        print(f"Distributed {classifiers_name} CLASSIFICATION")
        y_pred_avg = distributed_clf.predict(dataset_multi,fusion_method="average")
        print(classification_report(dataset_multi["ys"],y_pred_avg))




if __name__ == "__main__":
    main()




