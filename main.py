from helper import *

# =========================================
#       For Cancer Classification model
# =========================================
os.chdir('D:/project/practice/breastcancer/data/')
# [1] ResNet50 + SVM, LR
train = CancerClass_Process('./train.json', (224,224))
train_x, train_y = train.constructXY()
test = CancerClass_Process('./test.json', (224,224))
test_x, test_y = test.constructXY()

# Load CNN base model for extracting features
conv_base = LoadModel._ResNet50()
train_x = extractFeature(train_x, conv_base)

# Load SVM and LR Classifier
cls = CLS(train_x, train_y)
svm_classifier = cls.SVMClassifer('f1')
lr_classifier = cls.LRClassifer('f1')

# Evaluation
eval_svm = EvaluateModel(svm_classifier, test_x, test_y)
eval_svm.Metrics() # print score
eval_svm.ConfusionMatrix() # plot confusion matrix

eval_lr = EvaluateModel(lr_classifier, test_x, test_y)
eval_lr.Metrics() # print score
eval_lr.ConfusionMatrix() # plot confusion matrix

# [2] VGG16 + SVM, LR
# [3] VGG19 + SVM, LR
# [4] Xception + SVM, LR

# =========================================
#       For Cancer Classification model
# =========================================