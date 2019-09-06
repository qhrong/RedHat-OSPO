# These file is for Pseudo-Labeling

# Load in xlsm (cannot use csv, will generate bugs)
import pandas as pd
ls1 = pd.read_excel('RedHat3.xlsm')
ls2 = pd.read_excel('FirstSetRedHat.xlsm')
# Keep the email from non red hat list
nrh_ls = pd.concat([ls1["cmt_author_email"], ls2["cmt_author_email"]])

# Need to load in complete data
# import random
# n = 52812768 #number of records in file
# s = 30000 #desired sample size
# skip = sorted(random.sample(range(n), n-s))
# df_samp = pd.read_csv("repos-complete.csv", skiprows=skip)
# df_pl = data_transformer(df_samp)
# df_pl.to_csv("pl_data.csv")

# Load in sample data
df = pd.read_csv("sample_rh.csv")
len(df[df['author_affiliation']=='redhat']) #6987

# See how many observations are in non rh list
df[df['author_email'].isin(list(nrh_ls))] #2625

# Add committer email back to transformed data
df_tmp = pd.read_csv("transformed_sample.csv")
# Add a column to indicate the neighbor's affiliation
res = []
for ele in df_tmp['Unnamed: 0']:
    try:
        compare_tmp = df['committer_email'][df['committer_name'] == ele]
        res.append(compare_tmp.iloc[0])
    except:
        res.append('Na')
df_tmp['email'] = res

# Filter out labeled data
# Labeled as Red Hat
df_rh = df_tmp[df_tmp['affiliation'] == 1]

# Labeled as Non-RH (cross over with the two lists)
df_nrh = df_tmp[df_tmp['email'].isin(list(nrh_ls))]

print(df_rh) #1091 rows
print(df_nrh) #255 rows

# Use labeled set to train Naive Bayes and Penalized SVM
# Given the number of rows in rh df and nrh df, some methods should be used to deal with unbalanced problem
# Seperate x_train, x_test, y_train, y_test
from sklearn.model_selection import train_test_split
# Aggregate two sets
df_rh['y'] = 1
df_nrh['y'] = 0
df_pl = pd.concat([df_rh, df_nrh])
del df_pl['email']

# Detect and remove errors generated in transformation
def error_detector(df):
    import numpy as np
    a = []
    b = []
    for i in range(len(df.columns)):
        for j in range(len(df)):
            if type(df.iloc[j,i]) == str:
                a.append(j)
                b.append(i)

    for i in a:
        for j in b:
            try:
                df.iloc[i,j] = int(df.iloc[i,j])
            except:
                df.iloc[i,j] = 3000 #set a large number as rank for exception

    for ele in df.columns:
        if sum(np.isnan(df[ele])) > 0:
            np.nan_to_num(df[ele], 0)

    return df
error_detector(df_pl.loc[:,df_pl.columns!='y'])

# Try SMOTE and ADASYN with Complement Naive Bayes
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.naive_bayes import ComplementNB
import numpy as np
X, y = df_pl.loc[:, df_pl.columns != 'y'], df_pl['y']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
print(X_resampled.head(), y_resampled.head())
cnb = ComplementNB()
cnb.fit(X_resampled, y_resampled)
y_pred_cnb = cnb.predict(X_test)
y_pred_cnb = np.where(y_pred_cnb > 0.5, 1, 0)


X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
print(X_resampled.head(), y_resampled.head())
cnb = ComplementNB()
cnb.fit(X_resampled, y_resampled)
y_pred_adasyn = cnb.predict(X_test)
y_pred_adasyn = np.where(y_pred_adasyn > 0.5, 1, 0)

# Try penalized SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf', gamma=0.5, C=1, class_weight='balanced')
y_pred_svm = clf.fit(X_test)
y_pred_svm = np.where(y_pred_svm > 0.5, 1, 0)

# Plot three sets of result on ROC
# 1. cnb curve
from sklearn import metrics
import matplotlib.pyplot as plt
fpr1, tpr1, thresh1 = metrics.roc_curve(y_test, y_pred_cnb)
auc1 = metrics.roc_auc_score(y_test, y_pred_cnb)
plt.plot(fpr1,tpr1,label="CNB, AUC="+str(auc1))
# 2. adasyn curve
fpr2, tpr2, thresh2 = metrics.roc_curve(y_test, y_pred_adasyn)
auc2 = metrics.roc_auc_score(y_test, y_pred_adasyn)
plt.plot(fpr2,tpr2,label="ADASYN, AUC=" + str(auc2))
# 3. SVM curve
fpr3, tpr3, thresh3 = metrics.roc_curve(y_test, y_pred_svm)
auc3 = metrics.roc_auc_score(y_test, y_pred_svm)
plt.plot(fpr3,tpr3,label="Penalized SVM, AUC=" + str(auc3))

# Predict unlabeled data with trained model


# Retrain the two models with complete sample data


# Compare two experiments with ROC and Entropy Loss
