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
from sklearn.model_selection import GridSearchCV

# Packages for pre-trained model
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, Xception
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def extractFeature(X, conv_base):
    tic = time()
    print("Start extracting features ...")
    features = []
    for img in X:
        img = np.expand_dims(img, axis=0)
        features.append(conv_base.predict(img)[0])
    toc = time()   
    print(f"Time in process: {toc-tic:.2f} seconds")
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
    def _VGG16(self):
        conv_base = VGG16(weights='imagenet', input_tensor=Input(shape=(224,224,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('flatten').output)
        return model
    @staticmethod
    def _VGG19(self):
        conv_base = VGG19(weights='imagenet', input_tensor=Input(shape=(224,224,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('flatten').output)
        return model
    @staticmethod
    def _Xception(self):
        conv_base = Xception(weights='imagenet', input_tensor=Input(shape=(299,299,3)))
        model = Model(inputs = conv_base.input, outputs = conv_base.get_layer('avg_pool').output)
        return model

class CancerClass_Process:
    # Before runing this code,
    # make sure that working dir is ./breastcancer/data/
    def __init__(self, jsonpath, dims):
        self.dims = dims
        self.jsonpath = jsonpath
        with open(self.jsonpath) as f:
            self.dataDict = json.load(f)

        print("Start loading file ...")
        self.filename = []
        for cname, mag in self.dataDict.items():
            for k, v in mag.items():
                v = ['./'+cname+'/'+k+'/'+v for v in v]
                self.filename.extend(v)
        print("Done.")

    def constructXY(self):
        print("Start processing images ...")
        tic = time()
        Encoder = LabelEncoder()
        images = [cv2.imread(p) for p in self.filename]
        resized = [cv2.resize(i, self.dims) for i in images]
        X = np.array(resized, dtype="float")/255.0
        Y = [cname.split('_')[1] for cname in self.filename]
        Y = Encoder.fit_transform(Y)
        Y = Y.reshape((Y.shape[0],1))
        toc = time()
        print(f"Time in process: {toc-tic:.2f} seconds")
        del self.filename, self.dataDict, images, resized
        return X, Y

class CLS:
    def __init__(self, fX, Y):
        self.fX = fX
        self.Y = Y
    def SVMClassifer(self, metric): # Argument metric means metric of fine-tuning step by GridSearch
        # Parameter besed on this link:
        # https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
        # For more info: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        cls = svm.SVC() 
        params = {'C': [0.1, 1, 10, 100, 1000], 
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear','rbf']} 
        
        print("Start training LR classifier ...")
        tic = time()
        print("Fine-tuning step")
        grid = GridSearchCV(cls, params, cv=10, scoring=metric)
        print("Training step")
        grid.fit(self.fX, self.Y)
        print(f"Best parameters : {grid.best_params_}")
        print(f"Best {metric} = {grid.best_score_:.3f}")
        toc = time()
        print(f"Time in process: {toc-tic:.2f} seconds")
        self.bestParams = grid.best_params_
        return grid
    def LRClassifer(self, metric): # Argument metric means metric of fine-tuning step by GridSearch 
        # Parameter besed on this link:
        # https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
        # For more info: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        # Solver param: https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
        cls = LogisticRegression()
        params = {'penalty': ['l1','l2'], 
              'C': np.logspace(-4, 4, 20),
              'solver': ['liblinear']} 

        print("Start training LR classifier ...")
        tic = time()
        print("Fine-tuning step")
        grid = GridSearchCV(cls , params, cv=10, scoring=metric)
        print("Training step")
        grid.fit(self.fX, self.Y)
        print(f"Best parameters : {grid.best_params_}")
        print(f"Best {metric} = {grid.best_score_:.3f}")
        toc = time()
        print(f"Time in process: {toc-tic:.2f} seconds")
        self.bestParams = grid.best_params_
        return grid

class EvaluateModel:
    def __init__(self, model, X, Ytrue):
        self.model = model
        self.X = X
        self.Ytrue = Ytrue
        self.Ypred = model.predict(X)
    def Metrics(self):
        acc = accuracy_score(self.Ytrue, self.Ypred)
        prec = precision_score(self.Ytrue, self.Ypred)
        recall = recall_score(self.Ytrue, self.Ypred)
        f1 = f1_score(self.Ytrue, self.Ypred)
        print(f"Accuracy  | {acc:.3f}")
        print(f"Precision | {prec:.3f}")
        print(f"Recall    | {recall:.3f}")
        print(f"F1-Score  | {f1:.3f}")
        # return acc, prec, recall, f1
    def ConfusionMatrix(self):
        plot_confusion_matrix(self.model, self.X, self.Ytrue)
        # (tn, fp, fn, tp) = confusion_matrix(self.Ytrue, self.Ypred, labels=[0,1]).ravel()
        # return (tn, fp, fn, tp)

    
        