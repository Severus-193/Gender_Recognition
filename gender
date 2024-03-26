import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes = True)
%matplotlib inline
df = pd.read_csv("../input/voicegender/voice.csv")
print("Dataset Loaded..")
df.head()
df.dtypes
df[‘label’].value_counts()
df.columns
df.isnull().sum()
df.describe()
df.corr()
df[‘label’].hist()
plt.show()
df.hist(figsize=(10,10))
plt.show()
plt.show()
df.shape()
x = df.iloc[:,:20]
y = df['label']
###Label Encoding
def label_transform(x):
le = LabelEncoder()
Encoded_le = le.fit_transform(x)
return Encoded_le
catagoral_label=["label"]
for i in catagoral_label:
df[i] = label_transform(df[i])
df.head()
###Pre-processing pipeline with feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
numeric_features=['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='mean')),
('scaler', StandardScaler())])
categorical_features = []
categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
transformers=[
('num', numeric_transformer, numeric_features),
('cat', categorical_transformer, categorical_features)])
# Append classifier to preprocessing pipeline.
test = SelectKBest(score_func=f_classif, k=4)
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
('feature_selection', SelectKBest(score_func=f_classif, k=4)),
('classification', LogisticRegression())
])
df.head()
###Train and Split data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,
random_state=21)
print('Shape of Training Xs:{}'.format(x_train.shape))
print('Shape of Test Xs:{}'.format(x_test.shape))
print('Shape of Training y:{}'.format(y_train.shape))
print('Shape of Test y:{}'.format(y_test.shape))
Shape of Training Xs:(2534, 20)
Shape of Test Xs:(634, 20)
Shape of Training y:(2534,)
Shape of Test y:(634,)
###Accuracy Score
clf.fit(x_train, y_train)
print("model score on testing set: %.3f" % clf.score(x_test, y_test))
y_pred = clf.predict(x_test)
###Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, x, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores
.std()))
###Classification Report
from sklearn.metrics import classification_report
# print classification report
print(classification_report(y_test, y_pred))
###Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cnf_matrix
import itertools
def plot_confusion_matrix(cm, classes,
normalize=False,
title='Confusion matrix',
cmap=plt.cm.Blues):
"""
This function prints and plots the confusion matrix.
Normalization can be applied by setting `normalize=True`.
"""
if normalize:
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Normalized confusion matrix")
else:
print('Confusion matrix, without normalization')
print(cm)
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
plt.text(j, i, format(cm[i, j], fmt),
horizontalalignment="center",
color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1],
title='Confusion matrix, without normalization')
# With normalization
plt.figure()
plot_confusion_matrix(cnf_matrix, classes= [0,1], normalize=True,
title='Normalized confusion matrix')
plt.show()
###Regression Plot Meanfreq vs label
import seaborn as sns
a = df['meanfreq']
b= df['label']
sns.regplot(x=a, y=b, data=df, logistic=True, ci=None)
###Classification Accuracy
TP = cnf_matrix [0,0]
TN = cnf_matrix [1,1]
FP = cnf_matrix [0,1]
FN = cnf_matrix [1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
###Classification Error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
###True Positive Rate
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
###False Positive Rate
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
###Specificity
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
