# test different regression models to evaluate performance


# load packages
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error


%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 12)
matplotlib.style.use('ggplot')

# connect to postgres database
# define database name and username
dbname = 'campaigns_db'
username = 'tess'
engine = create_engine('postgres://%s@localhost/%s' % (username, dbname))
print engine.url

# create a database (if it doesn't exist)
if not database_exists(engine.url):
    create_database(engine.url)
# print(database_exists(engine.url))
############


# Grab the individual contributions in 2015 + 2016
# Querying the '2016 table' in the campaigns database
# connect:
con = None
con = psycopg2.connect(database=dbname, user=username)

# query
sql_query = """
SELECT name, entity_tp, occupation, employer, transaction_amt, date, "ZIP", "COUNTY" 
FROM "2016_contributions_table" 
WHERE entity_tp = 'IND';
"""
contribs = pd.read_sql_query(sql_query, con)
############


# load American Community Survey census data
acs = pd.read_csv('../04_data/acs_subset_byZipCode_update.csv',
                  index_col=0, converters={'Zipcode': lambda y: str(y)})
############


# Grab the zipcode <--> 8 cluster crosswalk file
# connect:
con = None
con = psycopg2.connect(database=dbname, user=username)

# query
sql_query = """
SELECT * FROM "zipcodes_to_8clusters";
"""
zip8clust = pd.read_sql_query(sql_query, con)
contribs_zip = contribs.groupby('ZIP').sum()
############

# merge data
# group transactions by zipcodes
contribs_zip = contribs.groupby('ZIP').sum()
# reallocate zipcode as column
contribs_zip['Zipcode'] = contribs_zip.index
# merge demographic data with contributions
acs_contribs = acs.merge(contribs_zip, left_on='Zipcode', right_on='Zipcode')
# merge demographic data with cluster assignments
acs_contribs_8clusters = acs_contribs.merge(
    zip8clust, left_on='Zipcode', right_on='Zipcode')
############


# normalize features
# impute missing data (NaNs) in full dataset
acs_imp_clust = preprocessing.Imputer(missing_values='NaN', strategy='mean',
                                      axis=0, verbose=0, copy=True).fit_transform(acs_contribs_8clusters.ix[:, 1:-3])

# recreate full DF
acs_imp_clust_df = pd.DataFrame(acs_imp_clust, columns=acs_contribs_8clusters.columns[
                                1:-3], index=acs_contribs_8clusters.index)
acs_imp_clust_df['Zipcode'] = acs_contribs_8clusters['Zipcode']
acs_imp_clust_df['Cluster8Assignment'] = acs_contribs_8clusters[
    'Cluster8Assignment']

# standardize the data to lie in range ~(0,1)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 1.01))
# .ix[:,:-2]: don't standardize the zipcodes or cluster assigments
acs_imp_clust_df_norm = min_max_scaler.fit_transform(
    acs_imp_clust_df.ix[:, :-2])
# full dataset:
Xy = pd.DataFrame(acs_imp_clust_df_norm, columns=acs_imp_clust_df.ix[
                  :, :-2].columns, index=acs_imp_clust_df.ix[:, :-2].index)
############


# standardize data using boxcox
Xy_bc = pd.DataFrame(acs_imp_clust_df_norm, columns=acs_imp_clust_df.ix[
                     :, :-2].columns, index=acs_imp_clust_df.ix[:, :-2].index)
Xy_bc['Cluster8Assignment'] = acs_imp_clust_df['Cluster8Assignment']
Xy_bc['sumDollars'] = acs_contribs['transaction_amt']
Xy_bc['logSumDollars'] = log(Xy_bc['sumDollars'])
Xy_bc_cutoff = Xy_bc[Xy_bc['logSumDollars'] > 0].dropna()

Xy_bc_cutoff_normed = pd.DataFrame()
for i in Xy_bc_cutoff.columns:
    Xy_bc_cutoff_normed[i] = preprocessing.scale(boxcox(Xy_bc_cutoff[i])[0])
############


# trying different regression models
def testTrainSplit(Xy_cutoff):

    # randomly shuffle rows in df
    Xy_cutoff_shuffle = Xy_cutoff.reindex(
        np.random.permutation(Xy_cutoff.index))

    # create features and response variables
    X = Xy_cutoff_shuffle.ix[:, :-1]
    y = Xy_cutoff_shuffle.ix[:, -1:]

    # 50 - 50 split into train and test
    n_samples = Xy_cutoff.shape[0]
    if n_samples % 2 == 0:
        X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
        X_test, y_test = X[n_samples / 2:], y[n_samples / 2:]
    else:
        X_train, y_train = X[:n_samples / 2], y[:n_samples / 2]
        X_test, y_test = X[n_samples / 2:-1], y[n_samples / 2:-1]

    return X_train, y_train, X_test, y_test


# perform linear regression
def linearReg(X_train, y_train, X_test, y_test):

    lm = LinearRegression()
    y_pred_lm = lm.fit(X_train, y_train).predict(X_test)
    r2_score_lm = r2_score(y_test, y_pred_lm)

    return lm, y_pred_lm, r2_score_lm


# perform Bayesian Ridge regression
def BRidgeReg(X_train, y_train, X_test, y_test, alpha=1.0):

    bridge = BayesianRidge(normalize=False, fit_intercept=True)
    y_pred_bridge = bridge.fit(X_train, y_train).predict(X_test)
    r2_score_bridge = r2_score(y_test, y_pred_bridge)

    return bridge, y_pred_bridge, r2_score_bridge

# perform Ridge regression


def RidgeReg(X_train, y_train, X_test, y_test, alpha=1.0):

    ridge = Ridge(alpha=alpha, normalize=False,
                  fit_intercept=True, max_iter=1000000)
    y_pred_ridge = ridge.fit(X_train, y_train).predict(X_test)
    r2_score_ridge = r2_score(y_test, y_pred_ridge)

    return ridge, y_pred_ridge, r2_score_ridge


# perform Random Forest Regression
def RandomForestReg(X_train, y_train, X_test, y_test):

    rf = RandomForestRegressor(n_estimators=10, max_features=10,
                               max_depth=1000, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
    y_pred_rf = rf.fit(X_train, y_train).predict(X_test)
    r2_score_rf = r2_score(y_test, y_pred_rf)

    return rf, y_pred_rf, r2_score_rf

# perform Kernel Ridge Regression


def kernelRidgeReg(X_train, y_train, X_test, y_test):

    kr = KernelRidge(kernel='rbf')
    y_pred_kr = kr.fit(X_train, y_train).predict(X_test)
    r2_score_kr = r2_score(y_test, y_pred_kr)

    return kr, y_pred_kr, r2_score_kr

# perform Gradient Boosting Regression


def GradBoostReg(X_train, y_train, X_test, y_test):
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    gbr = GradientBoostingRegressor(**params)
    y_pred_gbr = gbr.fit(X_train, y_train['logSumDollars']).predict(X_test)
    r2_score_gbr = r2_score(y_test['logSumDollars'], y_pred_gbr)

    return gbr, y_pred_gbr, r2_score_gbr

############


# train / test split
X_train, y_train, X_test, y_test = testTrainSplit(Xy_bc_cutoff_normed)
observed_y = y_test

# linear regression full model using boxcox scaled data:
lin, lin_pred, r2_score_lin = linearReg(X_train, y_train, X_test, y_test)
predictions_y = lin_pred
# print "r^2 on test data : %f" % r2_score_lin
#plt.scatter(lin_pred, y_test)
# returns R2 of 0.506
############

############
# ridge regression full model using boxcox scaled data:
ridge, ridge_pred, r2_score_ridge = RidgeReg(X_train, y_train, X_test, y_test)
predictions_y = ridge_pred

# print "r^2 on test data : %f" % r2_score_ridge
#plt.scatter(ridge_pred, y_test)
# returns R2 of 0.515
############


############
# random forest regression using boxcox scaled data:
rf, rf_pred, r2_score_rf = RandomForestReg(
    X_train, y_train['logSumDollars'], X_test, y_test['logSumDollars'])
predictions_y = rf_pred

# print "r^2 on test data : %f" % r2_score_rf
#plt.scatter(rf_pred, y_test)
# returns R2 of 0.513
############


############
# kernel ridge regression using boxcox scaled data:
kr, kr_pred, r2_score_kr = KernelRidgeReg(
    X_train, y_train['logSumDollars'], X_test, y_test['logSumDollars'])
predictions_y = kr_pred
# print "r^2 on test data : %f" % r2_score_kr
#plt.scatter(kr_pred, y_test)
# returns R2 of 0.588
############


############
# Gradient Boosting Regression using boxcox scaled data:
gbr, gbr_pred, r2_score_gbr = GradBoostReg(
    X_train, y_train['logSumDollars'], X_test, y_test['logSumDollars'])
predictions_y = gbr_pred
# print "r^2 on test data : %f" % r2_score_gbr
#plt.scatter(gbr_pred, y_test)
# returns R2 of 0.572


############
# Gradient Boosting Regression:
# predict expected contributions for each zip code,
# return priority scores
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = GradientBoostingRegressor(**params)
all_gbr, all_pred_GBR, r2_score_all = gbr.fit(
    X_train, y_train['logSumDollars']).predict(Xy_bc_cutoff_normed.ix[:, :-1])

priority = Xy_bc_cutoff_normed.ix[:, -1:]
priority['predicted'] = all_pred_GBR
priority['diff'] = exp(priority['predicted']) - exp(priority['logSumDollars'])
priority['Rank'] = priority['diff'].rank(ascending=True)
priority['Zipcode'] = acs_imp_clust_df['Zipcode']
priority = priority.drop(['zipcode'], axis=1)
priority.to_csv(
    '../04_data/prioritization_of_ZipCodes_byMoneyToBeRaised.csv', header=True)
############


############
# draw top 100 priority zip codes

gbr_out = y_test
gbr_out['predicted'] = y_pred_GBR
gbr_out['diff'] = exp(gbr_out['predicted']) - exp(gbr_out['logSumDollars'])

plt.hold(True)

circle = gbr_out['diff'] >= 186937.130865

# scatter non-targeted points in blue (c='b')
plt.scatter(exp(gbr_out['predicted'][~circle]), exp(
    gbr_out['logSumDollars'][~circle]), label='not target', c='b')
# scatter targeted points in organge
plt.scatter(exp(gbr_out['predicted'][circle]), exp(
    gbr_out['logSumDollars'][circle]), label='target', c='orange', s=80)

# add trend line
plt.plot(exp(gbr_out['predicted']), exp(
    gbr_out['predicted']), c='orange', linewidth=2)
# set scale to log
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(
    '../02_images/target_top100ZipCodes_GBR_predictions.pdf', format='pdf')
