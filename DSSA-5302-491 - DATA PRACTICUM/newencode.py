# %%
from pathlib import Path
import pandas as pd 
from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.manifold import TSNE
# this one sucks to install
# https://xgboost.readthedocs.io/en/latest/build.html
# https://anaconda.org/anaconda/py-xgboost
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, ShuffleSplit, KFold, StratifiedKFold
from sklearn import feature_selection
from sklearn import model_selection, metrics, feature_selection
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils import class_weight
import seaborn as sns
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from numpy.random import seed
seed(911)

#pd.set_option('display.max_columns', None)


#https://towardsdatascience.com/building-an-employee-churn-model-in-python-to-develop-a-strategic-retention-plan-57d5bd882c2d
#https://towardsdatascience.com/data-preprocessing-for-machine-learning-in-python-2d465f83f18c
#https://towardsdatascience.com/data-preprocessing-3cd01eefd438
#https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5
# %%
f = Path(__file__).parent / "../../Data/dssa_rentention_test.csv"

data = pd.read_csv(f, thousands=',')

f = Path(__file__).parent / "../../Data/dssa_course_test.csv"
# import raw CSV
crs_data = pd.read_csv(f, thousands=',')
dcol = ['SUNDAY','MONDAY','TUESDAY','WEDNESDAY','THURSDAY', 'FRIDAY','SATURDAY']
tcol = ['MORNING','AFTERNOON','NIGHT']

days = crs_data.groupby(['HSTU_ID','TERM_CODE'])[dcol].sum()
days = days.div(days.sum(axis=1), axis=0)

times = crs_data.groupby(['HSTU_ID','TERM_CODE'])[tcol].sum()
times = times.div(times.sum(axis=1), axis=0)

data = data.merge(days, how='left', left_on=['HSTU_ID', 'REG_TERM'], right_on=['HSTU_ID', 'TERM_CODE'])
data = data.merge(times, how='left', left_on=['HSTU_ID', 'REG_TERM'], right_on=['HSTU_ID', 'TERM_CODE'])
data[dcol + tcol] = data[dcol + tcol].fillna(0.0)

# we  want to drop 2020 bc covid
data = data[data['REG_TERM'] != 202020]
# drop columns that have lots of missing data
data = data.drop(['SPRIDEN_PIDM','HSTU_ID','PARENT_EDUC','EFC','INCOME',
'HOUSEHOLD','PARENT_EDUC','MINOR'], axis=1)

# remove rows with lots of NA's 
#data.isnull().sum()
# Check for NA Values
#df1 = data[data.isna().any(axis=1)]


data = data.dropna(subset=['SEX','CR_EARNED','CR_ATTEMPTED'])

# drop Summer terms
data = data[data['TERM_TYPE'] != 2]

# convert PoG
data['PoG'] = np.where((data['PERSISTENCE'] == 1) | (data['GRADUATION'] == 1), 1, 0)

# Calc Credit Difference
data['CR_DIFF'] = data['CR_ATTEMPTED'] - data['CR_EARNED']



#data['SEX'] = data['SEX'].fillna("NA")
#data[['CR_EARNED', 'CR_ATTEMPTED']] = data[['CR_EARNED', 'CR_ATTEMPTED']].fillna(0)
data['PREV_TERM_GPA'] = data['PREV_TERM_GPA'].fillna(4.0)
data['CUMUL_GPA'] = data['CUMUL_GPA'].fillna(4.0)
data['CURR_TERM_GPA'] = data['CURR_TERM_GPA'].fillna(4.0)
data['COHORT_CODE'] = data['COHORT_CODE'].fillna('IPTRF_NA')


data = data.drop(['CURR_TERM_GPA'], axis=1)
#data.SEX.value_counts()
data = data[data.SEX != 'N']

# Create DF for freshman and upperclassman 
data_fresh = data[data['CLASS_LEVEL'] == "FR"]
data_upper = data[data['CLASS_LEVEL'] != "FR"]

#data[['MAJOR', 'MINOR', 'RACE_ETH']] = data[['MAJOR', 'MINOR', 'RACE_ETH']].fillna("NA")
#data['HOUSEHOLD'] = data['HOUSEHOLD'].fillna("NA")
#data['INCOME'] = data['INCOME'].fillna("NA")
# shuffle data
data = utils.shuffle(data)

# %% 
le = preprocessing.LabelEncoder()

#%%
# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in data.columns[0:]:
    if data[col].dtype == 'object':
        if len(list(data[col].unique())) <= 2:
            print(col)
            le.fit(data[col])
            data[col] = le.transform(data[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))

#%%
# Start of OHE 
ohe_features = ['MAJOR', 'RACE_ETH','COHORT_CODE','CLASS_LEVEL']
ohe = preprocessing.OneHotEncoder()
ohe_feature_df = data[ohe_features]
onehot = ohe.fit_transform(ohe_feature_df)
onehotdf = pd.DataFrame.sparse.from_spmatrix(onehot)
onehotdf.columns = [ohe.get_feature_names(['MAJOR','RACE_ETH','COHORT_CODE','CLASS_LEVEL'])]
onehotdf = onehotdf.sparse.to_dense()

onehotdf.reset_index(drop=True, inplace=True)
data.reset_index(drop=True, inplace=True)
data = pd.concat([data, onehotdf], axis=1)
data = data.drop(ohe_features, axis=1)


pred_data = data[data['REG_TERM'] == 201980]
data.drop(data.loc[data['REG_TERM'] == 201980].index)


data = data.drop(['REG_TERM'], axis=1)
pred_data = pred_data.drop(['REG_TERM'], axis=1)
# convert rest of categorical variable into dummy
#data = pd.get_dummies(data, drop_first=True)


#%%

# Robust Scaler
# scaler = preprocessing.RobustScaler()
# data_col = list(data.columns)
# data_col.remove('GRADUATION')
# data_col.remove('PERSISTENCE')
# data_col.remove('PoG')
# for col in data_col:
#     data[col] = data[col].astype(float)
#     data[[col]] = scaler.fit_transform(data[[col]])
# data['PoG'] = pd.to_numeric(data['PoG'], downcast='float')
# data.head()

#%%
# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_col = list(data.columns)
data_col.remove('GRADUATION')
data_col.remove('PERSISTENCE')
data_col.remove('PoG')
for col in data_col:
    data[col] = data[col].astype(float)
    data[[col]] = scaler.fit_transform(data[[col]])
data['PoG'] = pd.to_numeric(data['PoG'], downcast='float')
data.head()

#%%
# assign the target to a new dataframe and convert it to a numerical feature
#df_target = df_HR[['PoG']].copy()
target = data['PoG'].copy()
pred_target = pred_data['PoG'].copy()


# #%%
# data['PoG'].iplot(kind='hist', xTitle='PoG',
#                          yTitle='count', title='PoG Distribution')

# # %%

# data['PoG'].value_counts()
#%%

# let's remove the target feature and redundant features from the dataset
pred_data.drop(['GRADUATION', 'PERSISTENCE','PoG'], axis=1, inplace=True)
data.drop(['GRADUATION', 'PERSISTENCE','PoG'], axis=1, inplace=True)
print('Size of Full dataset is: {}'.format(data.shape))

# #%%
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from statsmodels.tools.tools import add_constant

# vif = pd.DataFrame()
# df = add_constant(data)
# vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
# vif['features'] = df.columns

# %%
# Since we have class imbalance (i.e. more PoG with 1 than 0)
# let's use stratify=y to maintain the same ratio as in the training dataset when splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=7,
                                                    stratify=target)  
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)



# %% 
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_smote, y_smote = sm.fit_resample(X_train,y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

X_smote = np.asarray(X_smote)
y_smote= np.asarray(y_smote)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

#pred_data = np.asarray(pred_data)
#pred_target = np.asarray(pred_target)

#%%
# selection of algorithms to consider and set performance measure
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7,
                                                         class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier(
    n_estimators=100, random_state=7)))
#models.append(('SVM', SVC(gamma='auto', random_state=7)))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('Decision Tree Classifier',
#               DecisionTreeClassifier(random_state=7)))
#models.append(('Gaussian NB', GaussianNB()))
#models.append(('PCA', PCA(n_components=2, random_state=7)))
#models.append(('TSNE',TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)))



# PCA ANN TSNE NEED TO BE ADDED


#%%
acc_results = []
auc_results = []
names = []
# set table to table to populate with performance results
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 
       'Accuracy Mean', 'Accuracy STD']
df_results = pd.DataFrame(columns=col)
i = 0
# evaluate each model using cross-validation
for name, model in models:
    kfold = model_selection.StratifiedKFold(random_state=16)  # 10-fold cross-validation

    cv_acc_results = model_selection.cross_val_score(  # accuracy scoring
        model, X_smote, y_smote, cv=kfold, scoring='accuracy', verbose=True, n_jobs=-1)

    cv_auc_results = model_selection.cross_val_score(  # roc_auc scoring
        model, X_smote, y_smote, cv=kfold, scoring='roc_auc', verbose=True, n_jobs=-1)

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    names.append(name)
    df_results.loc[i] = [name,
                         round(cv_auc_results.mean()*100, 2),
                         round(cv_auc_results.std()*100, 2),
                         round(cv_acc_results.mean()*100, 2),
                         round(cv_acc_results.std()*100, 2)
                         ]
    i += 1
    print('done with',name,model)
df_results.sort_values(by=['ROC AUC Mean'], ascending=False)

# %%

tsne = TSNE(verbose=1, n_jobs=-1)
X_embedded = tsne.fit_transform(X_train)

# %%
palette = sns.color_palette("bright", 107380)

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=X_embedded[:,1], legend='full')

#%%

fig = plt.figure(figsize=(15, 7))
fig.suptitle('Algorithm Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc_results)
ax.set_xticklabels(names)
plt.show()


# %%
dft = data.copy()
pca_scores = PCA().fit_transform(dft)
df_pc = pd.DataFrame(pca_scores)
print('Explained variation per principal component: {}'.format(pca_scores.explained_variance_ratio_))
df = pd.DataFrame()
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 

#%%
tsne_em = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12, n_iter=1000, learning_rate=368, verbose=1, n_jobs=-1).fit_transform(df_pc.loc[:,0:49])

# %%

class general:
    def __init__(self):
        pass

    rand_colors = ('#a7414a', '#282726', '#6a8a82', '#a37c27', '#563838', '#0584f2', '#f28a30', '#f05837',
                   '#6465a5', '#00743f', '#be9063', '#de8cf0', '#888c46', '#c0334d', '#270101', '#8d2f23',
                   '#ee6c81', '#65734b', '#14325c', '#704307', '#b5b3be', '#f67280', '#ffd082', '#ffd800',
                   '#ad62aa', '#21bf73', '#a0855b', '#5edfff', '#08ffc8', '#ca3e47', '#c9753d', '#6c5ce7')

    def get_figure(show, r, figtype, fig_name):
        if show:
            plt.show()
        else:
            plt.savefig(fig_name+'.'+figtype, format=figtype, bbox_inches='tight', dpi=r)
        plt.close()

    def axis_labels(x, y, axlabelfontsize=None, axlabelfontname=None):
        plt.xlabel(x, fontsize=axlabelfontsize, fontname=axlabelfontname)
        plt.ylabel(y, fontsize=axlabelfontsize, fontname=axlabelfontname)
        # plt.xticks(fontsize=9, fontname="sans-serif")
        # plt.yticks(fontsize=9, fontname="sans-serif")

    def axis_ticks(xlm=None, ylm=None, axtickfontsize=None, axtickfontname=None, ar=None):
        if xlm:
            plt.xlim(left=xlm[0], right=xlm[1])
            plt.xticks(np.arange(xlm[0], xlm[1], xlm[2]),  fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        else:
            plt.xticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

        if ylm:
            plt.ylim(bottom=ylm[0], top=ylm[1])
            plt.yticks(np.arange(ylm[0], ylm[1], ylm[2]),  fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)
        else:
            plt.yticks(fontsize=axtickfontsize, rotation=ar, fontname=axtickfontname)

    def depr_mes(func_name):
        print("This function is deprecated. Please use", func_name )
        print("Read docs at https://reneshbedre.github.io/blog/howtoinstall.html")

    def check_for_nonnumeric(pd_series=None):
        if pd.to_numeric(pd_series, errors='coerce').isna().sum() == 0:
            return 0
        else:
            return 1


def tsneplot(score=None, axlabelfontsize=9, axlabelfontname="Arial", figtype='png', r=300, show=False,
            markerdot="o", dotsize=6, valphadot=1, colordot='#4a4e4d', colorlist=None, legendpos='best',
            figname='tsne_2d', dim=(6, 4), legendanchor=None):
    assert score is not None, "score are missing"
    plt.subplots(figsize=dim)
    if colorlist is not None:
        unique_class = set(colorlist)
        # color_dict = dict()
        assign_values = {col: i for i, col in enumerate(unique_class)}
        color_result_num = [assign_values[i] for i in colorlist]
        if colordot and isinstance(colordot, (tuple, list)):
            colour_map = ListedColormap(colordot)
            s = plt.scatter(score[:, 0], score[:, 1], c=color_result_num, cmap=colour_map,
                            s=dotsize, alpha=valphadot, marker=markerdot)
            plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos,
                            bbox_to_anchor=legendanchor)
        elif colordot and not isinstance(colordot, (tuple, list)):
            s = plt.scatter(score[:, 0], score[:, 1], c=color_result_num,
                            s=dotsize, alpha=valphadot, marker=markerdot)
            plt.legend(handles=s.legend_elements()[0], labels=list(unique_class), loc=legendpos,
                        bbox_to_anchor=legendanchor)
    else:
        plt.scatter(score[:, 0], score[:, 1], color=colordot,
                    s=dotsize, alpha=valphadot, marker=markerdot)
    plt.xlabel("t-SNE-1", fontsize=axlabelfontsize, fontname=axlabelfontname)
    plt.ylabel("t-SNE-2", fontsize=axlabelfontsize, fontname=axlabelfontname)
    general.get_figure(show, r, figtype, figname)

#%%

tsneplot(score=tsne_em)

#%%
from sklearn.cluster import DBSCAN
from matplotlib.colors import ListedColormap
get_clusters = DBSCAN(eps=3, min_samples=10).fit_predict(tsne_em)
set(get_clusters)

tsneplot(score=tsne_em, colorlist=get_clusters, 
    colordot=('#b0413e','#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#4381c1', '#736ced', '#631a86', '#de541e', '#022b3a', '#000000'), 
    legendpos='upper right', legendanchor=(1.15, 1))


#%%
# Regular K Fold
#kfold = model_selection.KFold(n_splits=10, random_state=7)
# Stratified K fold 
kfold = model_selection.StratifiedKFold(random_state=16)

modelCV = LogisticRegression(solver='liblinear',
                             class_weight="balanced", 
                             random_state=7)
scoring = 'roc_auc'
resultsauc = model_selection.cross_val_score(
    modelCV, X_smote, y_smote, cv=kfold, scoring=scoring)
resultsacc = model_selection.cross_val_score(  # accuracy scoring
        modelCV, X_smote, y_smote, cv=kfold, scoring='accuracy', verbose=True, n_jobs=-1)


print("AUC score (STD): %.2f (%.2f)" % (resultsauc.mean(), resultsauc.std()))
print("acc score : %.2f (%.2f)" % (resultsacc.mean(), resultsacc.std()))


#%%
param_grid = {'C': np.arange(1e-03, 2, 0.01)} # hyper-parameter list to fine-tune
log_gs = GridSearchCV(LogisticRegression(solver='liblinear', # setting GridSearchCV
                                         class_weight="balanced", 
                                         random_state=7),
                      iid=True,
                      n_jobs=-1,
                      return_train_score=True,
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=2,
                      verbose=True)

log_grid = log_gs.fit(X_train, y_train)
log_opt = log_grid.best_estimator_
results = log_gs.cv_results_

print('='*20)
print("best params: " + str(log_gs.best_estimator_))
print("best params: " + str(log_gs.best_params_))
print('best score:', log_gs.best_score_)
print('='*20)

#%%
## Confusion Matrix
cnf_matrix = metrics.confusion_matrix(y_test, log_opt.predict(X_test))
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#%%
print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(log_opt.score(pred_data, pred_target)*100))


#%%
log_opt.fit(X_smote, y_smote)
print(classification_report(y_test, log_opt.predict(X_test)))

# %%
log_opt.fit(X_smote, y_smote) # fit optimised model to the training data
probs = log_opt.predict_proba(X_test) # predict probabilities
probs = probs[:, 1] # 
logit_roc_auc = roc_auc_score(y_test, probs) # calculate AUC score using test dataset
print('AUC score: %.3f' % logit_roc_auc)

# %%
rf_classifier = RandomForestClassifier(class_weight = "balanced",
                                       random_state=7)
param_grid = {'n_estimators': [50, 75, 100, 125, 150, 175],
              'min_samples_split':[2,4,6,8,10],
              'min_samples_leaf': [1, 2, 3, 4],
              'max_depth': [5, 10, 15, 20, 25]}

grid_obj = GridSearchCV(rf_classifier,
                        iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=2,
                        verbose=True,
                        n_jobs=-1)

grid_fit = grid_obj.fit(X_smote, y_smote)
rf_opt = grid_fit.best_estimator_

print('='*20)
print("best params: " + str(grid_obj.best_estimator_))
print("best params: " + str(grid_obj.best_params_))
print('best score:', grid_obj.best_score_)
print('='*20)

#%%
forest = RandomForestClassifier(max_depth=6, random_state=0)
forest.fit(X_smote, y_smote)

y_pred = forest.predict(pred_data)
y_prob = forest.predict_proba(pred_data)[:, 1]

print(confusion_matrix(pred_target, y_pred))
print(cross_val_score(  # accuracy scoring
        forest, X_smote, y_smote, scoring='accuracy', n_jobs=-1))

print(roc_auc_score(pred_target, y_prob))


model_fpr, model_tpr, _ = roc_curve(pred_target, y_prob)

plt.plot(model_fpr, model_tpr, 'r', label = 'model')

fi = pd.DataFrame({'feature': list(data.columns),
                   'importance': forest.feature_importances_}).\
                    sort_values('importance', ascending = False)

print(fi)



#%%
cnf_matrix = confusion_matrix(pred_target, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# %%
importances = rf_opt.feature_importances_
indices = np.argsort(importances)[::-1] # Sort feature importances in descending order
names = [X_smote.columns[i] for i in indices] # Rearrange feature names so they match the sorted feature importances
plt.figure(figsize=(15, 7)) # Create plot
plt.title("Feature Importance") # Create plot title
plt.bar(range(X_smote.shape[1]), importances[indices], ) # Add bars
plt.xticks(range(X_smote.shape[1]), names, rotation=90) # Add feature names as x-axis labels
plt.tick_params(axis='x', which='major', labelsize=4)
#ax.set_xticklabels(labels = names, rotation = (45), fontsize = 2, va='bottom', ha='left')
plt.show() # Show plot


#%%
from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

#%%
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=0)
# X_res, y_res = sm.fit_resample(X_train,y_train)

# X_test = np.asarray(X_test)
# y_test = np.asarray(y_test)

# X_res = np.asarray(X_res)
# y_res= np.asarray(y_res)


# %%
model = keras.Sequential()
model.add(Dense(16, input_dim=len(data_col), activation="relu"))
model.add(Dropout(.2))
model.add(Dense(8, activation="relu"))
model.add(Dropout(.2))
model.add(Dense(1, activation="sigmoid"))
model_config = model.get_config()

#weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#weight = {i : weight[i] for i in range(2)}

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

standard_model = keras.Sequential.from_config(model_config)
standard_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mse', 'mae', 'mape'])
standard_model.fit(
    X_smote, y_smote, 
    validation_data=(X_test, y_test), 
    epochs=10, 
    batch_size=64, 
    verbose=0,
    class_weight=class_weight.compute_class_weight('balanced', np.unique(y_smote), y_smote),
    callbacks=[es]
)



#%%
df1 = pd.DataFrame(np.array([[32,495],[38,8257]]))
blah = standard_model.evaluate(pred_data.values, pred_target.values, verbose=0)
print('Test loss:', blah[0])
print('Test accuracy:', blah[1])
y_pred_class = model.predict_classes(pred_data.values, verbose=0)
cnf_matrix = confusion_matrix(y_pred_class, pred_target.values)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
sns.heatmap(df1, annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


#%%
def run_metrics(model, X_test, y_test,pred_target):
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    # plt.plot(model.history.history['mse'])
    # plt.plot(model.history.history['mae'])
    # plt.plot(model.history.history['mape'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
     
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # assumes .5 as the round value
    y_pred_class = model.predict_classes(X_smote, verbose=0)

    # get number of epochs ran for
    epoch = len(model.history.history['loss'])

    print('=========================')
    print(f'Confusion Matrix {epoch} epoch')
    print('=========================')
    cm = confusion_matrix(y_test, y_pred_class)
    print('True negatives: ',  cm[0,0])
    print('False negatives: ', cm[1,0])
    print('False positives: ', cm[0,1])
    print('True positives: ',  cm[1,1]) 

    print(cm)

    # precision = tp / tp + fp
    print(f"Precision: {cm[1,1] / (cm[1,1] + cm[0,1])}")
    # recall = tp / tp + fn
    print(f"Recall: {cm[1,1] / (cm[1,1] + cm[1,0])}")

    # precision = tn / tn + fn
    print(f"Neg Precision: {cm[0,0] / (cm[0,0] + cm[1,0])}")
    # recall = tn / tn + fp
    print(f"Neg Recall: {cm[0,0] / (cm[0,0] + cm[0,1])}")


    #cnf_matrix = confusion_matrix(pred_target, y_pred)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


run_metrics(standard_model, X_smote, y_smote, pred_target)


# %%

# %%
