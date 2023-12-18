#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

import sklearn
import sklearn.compose
import sklearn.dummy
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

import scipy.ndimage

from sklearn.neural_network import MLPClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")
parser.add_argument("--mlps", default=5, type=int, help="Number of MLPs to train")

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


def augment_image(image):
    image = image.reshape(28, 28)
    
    rotated = scipy.ndimage.rotate(image, np.random.uniform(-15, 15), reshape=False, mode='reflect')
    
    zoom_factor = np.random.uniform(0.9, 1.1)
    zoomed = scipy.ndimage.zoom(rotated, zoom_factor)
    
    if zoom_factor > 1:
        crop_size = (zoomed.shape[0] - 28) // 2
        zoomed = zoomed[crop_size:crop_size+28, crop_size:crop_size+28]
    else:
        pad_size = (28 - zoomed.shape[0]) // 2
        zoomed = np.pad(zoomed, ((pad_size, 28 - zoomed.shape[0] - pad_size), 
                                 (pad_size, 28 - zoomed.shape[1] - pad_size)), mode='constant')
    
    return zoomed.reshape(-1)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        data = train.data

        mean_px = data.mean().astype(np.float32)
        std_px = data.std().astype(np.float32)
        data = (data - mean_px)/(std_px)

        target = train.target

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size = 0.2, random_state = 42)


        # pipeline = sklearn.pipeline.Pipeline([
        #     ("scaler", sklearn.preprocessing.StandardScaler()),
        #     ("pca", sklearn.decomposition.PCA(n_components=0.8))
        # ])

        # train_data_transformed = pipeline.fit_transform(train_data)
        # test_data_transformed = pipeline.transform(test_data)

        # mlp = MLPClassifier(
        #     tol=1e-4, verbose=1, alpha=0.001, hidden_layer_sizes=(500), max_iter=1000, random_state=42)
        
        # param_grid = {
        #     'hidden_layer_sizes': [(500,), (300, 200), (100, 100, 100)],
        #     'alpha': [0.0001, 0.001, 0.01],
        #     'learning_rate_init': [0.001, 0.01]
        # }

        # grid_search = sklearn.model_selection.GridSearchCV(mlp, param_grid, cv=3, verbose=2, n_jobs=-1)

        # grid_search.fit(train_data_transformed, train_target)

        # print("Best parameters found: ", grid_search.best_params_)

        # model = grid_search.best_estimator_
        # pred = model.predict(test_data_transformed)
        # print(f"Accuracy: {sklearn.metrics.accuracy_score(test_target, pred)}")

        # results = grid_search.cv_results_

        # sorted_indices = np.argsort(results['mean_test_score'])[::-1]

        # print("Rank, Mean Test Score, Parameters")
        # for rank, idx in enumerate(sorted_indices, start=1):
        #     print(f"{rank}, {results['mean_test_score'][idx]:.4f}, {results['params'][idx]}")

        alphas = [0.01, 0.0001, 0.0001, 0.001, 0.01]
        layers_sizes = [(300, 200), (500,), (300, 200), (300, 200), (500,)]

        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("pca", sklearn.decomposition.PCA(n_components=0.8)),
            ("MLPS", sklearn.ensemble.VotingClassifier([
                (f"MLP{i}", MLPClassifier(
                    tol=1e-4, verbose=1, alpha=alphas[i], hidden_layer_sizes=layers_sizes[i], max_iter=10, random_state=np.exp(i).astype(np.int32)))
                for i in range(args.mlps)
            ], voting="soft"))
        ])

        augmented_train_data = [augment_image(img) for img in train_data]

        combined_train_data = np.vstack((train_data, augmented_train_data))

        combined_train_target = np.hstack((train_target, train_target))

        model.fit(combined_train_data, combined_train_target)
        print(f"Accuracy: {sklearn.metrics.accuracy_score(test_target, model.predict_proba(test_data))}")

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained `MLPClassifier` is in the `mlp` variable.
        for mlp in model["MLPS"].estimators_:
            mlp._optimizer = None
            for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
            for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)


        pickle_out = open("classifier.pkl", "wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
