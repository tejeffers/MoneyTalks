# Hierarchical clustering of zipcodes

# load packages
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage

############
# prepare postgres database
%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 12)
matplotlib.style.use('ggplot')

# define database name and username
dbname = 'campaigns_db'
username = 'tess'
engine = create_engine('postgres://%s@localhost/%s' % (username, dbname))
print engine.url

# create a database (if it doesn't exist)
if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))
############


############
# Querying the '2016 table' in the campaigns database
# connect:
con = None
con = psycopg2.connect(database=dbname, user=username)

# query
sql_query = """
SELECT name, entity_tp, occupation, employer, transaction_amt, date, "ZIP", "COUNTY" FROM "2016_contributions_table" WHERE entity_tp = 'IND';
"""
contribs = pd.read_sql_query(sql_query, con)
############


############
# load American Community Survey census data
acs = pd.read_csv('../04_data/acs_subset_byZipCode_update.csv',
                  index_col=0, converters={'Zipcode': lambda y: str(y)})
# group transactions by zipcodes
contribs_zip = contribs.groupby('ZIP').median()
# reallocate zipcode as column
contribs_zip['Zipcode'] = contribs_zip.index
# merge acs with contributions
acs_contribs = acs.merge(contribs_zip, left_on='Zipcode', right_on='Zipcode')
############


############
# impute missing data (NaNs) in full dataset
acs_imp = preprocessing.Imputer(missing_values='NaN', strategy='mean',
                                axis=0, verbose=0, copy=True).fit_transform(acs_contribs.ix[:, 1:-1])
# cleanup after numpy conversion
acs_imp_df = pd.DataFrame(acs_imp, columns=acs_contribs.columns[
                          1:-1], index=acs_contribs.index)
acs_imp_df['Zipcode'] = acs_contribs['Zipcode']

# standardize the data to lie in range (0,1)
min_max_scaler = preprocessing.MinMaxScaler()
acs_imp_df_norm = min_max_scaler.fit_transform(acs_imp_df.ix[:, :-1])

# calculate linkage using ward's distance
Z = linkage(acs_imp_df_norm, 'ward')


############
# plot a truncated dendrogram with 8 clusters
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=8,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
# plt.show()
plt.savefig('..04_images/hierarchical_8Nodes.png', format='pdf')
############


############
# retrieve cluster assignments
assignments8 = fcluster(Z, 25, 'distance')
# save assignments <--> zip code crosswalk
zipcluster = pd.DataFrame([acs_imp_df['Zipcode'], assignments8])
zipcluster = zipcluster.T
zipcluster.rename(columns={'Unnamed 0': 'Cluster8Assignment'}, inplace=True)
zipcluster.to_csv(
    '../04_data/zipCodes_imputed_normalized_hierarchical_8Nodes_clusterAssignments.csv', header=True)

# also add to postgres
zipcluster.to_sql('zipcodes_to_8clusters', engine, if_exists='replace')
############


###########
# identify what group of people is in each cluster:
dems_groupByClusters = Xy_cutoff.groupby('Cluster8Assignment').median()
dems_groupByClusters.to_csv(
    '..04_data/Demographics_groupedByClusters_MedianValues.csv', header=True)
