from helper import *
from sklearn.metrics import plot_confusion_matrix, classification_report ,accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

os.chdir('D:/project/practice/breastcancer/data/')

# Load and read images
train = CancerClass_Process('./train.json')
train_x, train_y = train.constructXY((224,224))
test = CancerClass_Process('./test.json')
test_x, test_y = test.constructXY((224,224))

# ===============================================
#                   ResNet50
# ===============================================
conv_base = LoadModel._ResNet50()
train_x = extractFeature(train_x, conv_base)
test_x = extractFeature(test_x, conv_base)


cls = CLS(train_x, train_y)
# SVM
svm_classifier = cls.SVMClassifer(param_tuning=True)
svm_eval = EvaluateModel(svm_classifier, test_x, test_y)
svm_eval.metrics() # print score

# LR
lr_classifier = cls.LRClassifer(param_tuning=False)
lr_eval = EvaluateModel(lr_classifier, test_x, test_y)
lr_eval.metrics() # print score

# ===============================================
#                    VGG16
# ===============================================
conv_base = LoadModel._VGG16()
train_x = extractFeature(train_x, conv_base)
test_x = extractFeature(test_x, conv_base)


cls = CLS(train_x, train_y)
# SVM
svm_classifier = cls.SVMClassifer(param_tuning=True)
svm_eval = EvaluateModel(svm_classifier, test_x, test_y)
svm_eval.metrics() # print score

# LR
lr_classifier = cls.LRClassifer(param_tuning=True)
lr_eval = EvaluateModel(lr_classifier, test_x, test_y)
lr_eval.metrics() # print score

# ===============================================
#                    VGG19
# ===============================================
conv_base = LoadModel._VGG19()
train_x = extractFeature(train_x, conv_base)
test_x = extractFeature(test_x, conv_base)


cls = CLS(train_x, train_y)
# SVM
svm_classifier = cls.SVMClassifer(param_tuning=True)
svm_eval = EvaluateModel(svm_classifier, test_x, test_y)
svm_eval.metrics() # print score


# LR
lr_classifier = cls.LRClassifer(param_tuning=True)
lr_eval = EvaluateModel(lr_classifier, test_x, test_y)
lr_eval.metrics() # print score


# ===============================================
#                  * Xception *
# ===============================================
# Load and read images to construct new size
train = CancerClass_Process('./train.json')
train_x, train_y = train.constructXY((299,299))
test = CancerClass_Process('./test.json')
test_x, test_y = test.constructXY((299,299))

conv_base = LoadModel._Xception()
train_x = extractFeature(train_x, conv_base)
test_x = extractFeature(test_x, conv_base)


cls = CLS(train_x, train_y)
# SVM
svm_classifier = cls.SVMClassifer(param_tuning=True)
svm_eval = EvaluateModel(svm_classifier, test_x, test_y)
svm_eval.metrics() # print score


# LR
lr_classifier = cls.LRClassifer(param_tuning=True)
lr_eval = EvaluateModel(lr_classifier, test_x, test_y)
lr_eval.metrics() # print score
