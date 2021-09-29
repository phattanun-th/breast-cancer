import json
import os 
import numpy as np
import cv2
from time import time
from tqdm.notebook import tqdm

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, classification_report ,accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

# Packages for pre-trained model
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, Xception
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def extractFeature(X, conv_base):
    tic = time()
    print("[ Start extracting features ]")
    features = []
    for img in tqdm(X):
        img = np.expand_dims(img, axis=0)
        features.append(conv_base.predict(img)[0])
    toc = time()   
    print(f"Process time: {toc-tic:.2f} seconds\n")
    return np.array(features)


class LoadModel:
    # More info about model requirement:
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications
    @staticmethod
    def _ResNet50():
        conv_base = ResNet50(weights='imagenet' , input_tensor=Input(shape=(224,224,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('avg_pool').output)
        return model
    @staticmethod
    def _VGG16():
        conv_base = VGG16(weights='imagenet', input_tensor=Input(shape=(224,224,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('flatten').output)
        return model
    @staticmethod
    def _VGG19():
        conv_base = VGG19(weights='imagenet', input_tensor=Input(shape=(224,224,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('flatten').output)
        return model
    @staticmethod
    def _Xception():
        conv_base = Xception(weights='imagenet', input_tensor=Input(shape=(299,299,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('avg_pool').output)
        return model

class CancerClass_Process:
    # Before runing this code,
    # make sure that working dir is ./breastcancer/data/
    def __init__(self, jsonpath):
        self.jsonpath = jsonpath
        with open(self.jsonpath) as f:
            dataDict = json.load(f)

        self.filename = []
        for cname, mag in dataDict.items():
            for k, v in mag.items():
                v = ['./'+cname+'/'+k+'/'+v for v in v]
                self.filename.extend(v)
        del dataDict


    def constructXY(self, dims):
        print("[ Start processing images ]")
        tic = time()
        Encoder = LabelEncoder()
        X = [cv2.imread(f) for f in self.filename]
        X = [cv2.resize(im, dims) for im in X]
        X = np.array(X, dtype="float")/255.0
        Y = [cname.split('_')[1] for cname in self.filename]
        Y = Encoder.fit_transform(Y)
        # Y.reshape((-1,1)) to reshpe from (n,) to (n,1)
        toc = time()
        print(f"Process time: {toc-tic:.2f} seconds\n")
        del self.filename
        return X, Y

class CLS:
    def __init__(self, fX, Y):
        self.fX = fX
        self.Y = Y
    def SVMClassifer(self, param_tuning): # Argument metric means metric of fine-tuning step by GridSearch
        # Parameter besed on this link:
        # https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
        # For more info: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        print("[ Training SVM classifier ]")
        cls = svm.SVC() 
        if param_tuning :
            tic = time()
            print("find the best parameters ...")
            params = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear','rbf']} 
            grid = GridSearchCV(cls, params, cv=10, scoring='f1')
            grid.fit(self.fX, self.Y)
            print(f"Best parameters : {grid.best_params_}")
            print(f"Best F1-score = {grid.best_score_:.3f}")
            toc = time()
            print(f"Process time: {toc-tic:.2f} seconds\n")
            self.bestParams = grid.best_params_
            return grid
        else:
            cls.fit(self.fX, self.Y)
            print("Done")
            return cls

    def LRClassifer(self, param_tuning): # Argument metric means metric of fine-tuning step by GridSearch 
        # Parameter besed on this link:
        # https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
        # For more info: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        # Solver param: https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
        print("[ Training LR classifier ]")
        cls = LogisticRegression(max_iter=10000)
        if param_tuning :
            tic = time()
            print("find the best parameters ...")
            params = {'penalty': ['l1','l2'], 
              'C': np.logspace(-4, 4, 20),
              'solver': ['liblinear']} 
            grid = GridSearchCV(cls , params, cv=10, scoring='f1')
            grid.fit(self.fX, self.Y)
            print(f"Best parameters : {grid.best_params_}")
            print(f"Best F1-score = {grid.best_score_:.3f}")
            toc = time()
            print(f"Process time: {toc-tic:.2f} seconds\n")
            self.bestParams = grid.best_params_
            return grid
        else:
            cls.fit(self.fX, self.Y)
            print("Done")
            return cls            

class EvaluateModel:
    def __init__(self, model, X, Ytrue):
        self.model = model
        self.X = X
        self.Ytrue = Ytrue
        self.Ypred = model.predict(self.X)
    def metrics(self):
        cm = confusion_matrix(self.Ytrue, self.Ypred, labels=[0,1]).ravel()
        acc = accuracy_score(self.Ytrue, self.Ypred)
        prec = precision_score(self.Ytrue, self.Ypred)
        recall = recall_score(self.Ytrue, self.Ypred)
        f1 = f1_score(self.Ytrue, self.Ypred)
        print(f"Accuracy  | {acc:.3f}")
        print(f"Precision | {prec:.3f}")
        print(f"Recall    | {recall:.3f}")
        print(f"F1-Score  | {f1:.3f}")
        print(f"TN, FP, FN, TP = {cm}")


    
        