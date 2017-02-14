# preprocess dataframe for website

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
import folium
import json
import os

%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 12)
matplotlib.style.use('ggplot')

# load datasets for merging
data_path = '../04_data/'

# return only contributions from individuals (not corporations)
contribs = pd.read_csv(data_path + 'contributions16_reduced.csv',
                       index_col=0, converters={'ZIP': lambda y: str(y)})

# map between zipcode and lon/lat (to add data to map)
locs = pd.read_csv(data_path + 'zips_lonlats.csv',
                   converters={'ZIP': lambda y: str(y)})

# add cluster assignments from hierarchical geodemographic clustering
clusts = pd.read_csv(data_path + 'zipCodes_imputed_normalized_hierarchical_8Nodes_clusterAssignments.csv',
                     index_col=False, converters={'Zipcode': lambda y: str(y)})

# add priority score for zipcode targetting
priority = pd.read_csv(data_path + 'prioritization_of_ZipCodes_byMoneyToBeRaised.csv',
                       index_col=0, converters={'Zipcode': lambda y: str(y)})

############
# some functions here to turn continuous variables into categorical


def getTransactionSize(transaction_amt):
    '''Convert individual transaction amounts into categories'''
    if transaction_amt <= 20:
        return '<= $20'
    elif transaction_amt <= 100:
        return '<= $100'
    elif transaction_amt <= 1000:
        return '<= $1000'
    elif transaction_amt <= 5000:
        return '<= $5000'
    else:
        return ' > $5000'


def addToDic(dic):
    '''Convert job entry into occupation types'''
    legal = ['legal', 'law', 'lawyer', 'esquire', 'attorney', 'counsel',
             'paralegal', 'mediator', 'judge', 'court', 'courtroom', 'patent']
    student = ['student']
    government = ['government', 'politician', 'legislator', 'senator', 'representative',
                  'lobby', 'lobbyist', 'councilwoman', 'councilman', 'state', 'federal']
    education = ['professor', 'adjunct', 'faculty', 'teacher',
                 'school', 'university', 'education', 'teaching', 'learning']
    finance = ['finance', 'hedge', 'cpa', 'financial',
               'stock', 'portfolio', 'mutual', 'investor', 'fin']
    outdoors = ['farmer', 'contstruction', 'wildlife',
                'dnr', 'fish', 'hunt', 'hunting', 'fishing']
    business = ['business', 'director', 'planner', 'manager', 'officer', 'president', 'dir',
                'vp', 'pres', 'cto', 'ceo', 'coo', 'businessman', 'businesswoman', 'coordinator']
    homemaker = ['homemaker', 'house wife', 'housewife',
                 'stay', 'mom', 'mother', 'househusband']
    technology = ['engineer', 'technology', 'programmer', 'r&d', 'scientist',
                  'software', 'analyst', 'tech', 'data', 'website', 'site', ]
    retired = ['retired']
    medical = ['medical', 'doctor', 'health', 'md', 'dvm', 'rn', 'crna', 'optometrist', 'physician',
               'surgeon', 'veterinarian', 'nurse', 'technician', 'dentist', 'pharmacist', 'hygienist', 'technologist']
    creative = ['writer', 'director', 'museum', 'graphic', 'artist', 'art', 'design', 'poet', 'jewelry', 'designer', 'film', 'editor', 'filmmaker',
                'screenwriter', 'landscape', 'songwriter', 'seamstress', 'tv', 'sculptor', 'architect', 'creative', 'actress', 'musician', 'actor', 'advertising']

    job_list = [legal, student, government, education, finance, outdoors,
                business, homemaker, technology, retired, medical, creative]
    job_titles = ['legal', 'student', 'government', 'education', 'finance', 'outdoors',
                  'business', 'homemaker', 'technology', 'retired', 'medical', 'creative']

    index = 0
    for job in job_list:
        for item in job:
            dic[item] = job_titles[index]
        index += 1
    return dic

j = {}
jobs = addToDic(j)


def getJobClass(occupation):
    '''attach job type'''
    words = str(occupation).split()
    # for each word in the line:
    for word in words:
        word = word.strip().lower()
        if word in jobs:
            return jobs[word]
    else:
        return 'Not Listed'


def renameClusters(cluster):
    '''Attach names of clusters'''
    if cluster == 1:
        return 'Young Kids, Multi-racial, Low Density'
    elif cluster == 2:
        return 'Hispanic, non citizens, low income'
    elif cluster == 3:
        return 'Low Income, African American'
    elif cluster == 4:
        return 'Wealthy Urbanites (Renters)'
    elif cluster == 5:
        return 'Families, Low Density'
    elif cluster == 6:
        return 'White, Middle income Single Family'
    elif cluster == 7:
        return 'Suburban, White, Wealthy'
    elif cluster == 8:
        return 'Homeowners, Families'
    else:
        return 'None'

# merge datasets
# attach new columns
m1 = contribs.merge(clusts, left_on='ZIP', right_on='Zipcode')
m2 = m1.merge(locs, left_on='Zipcode', right_on='ZIP')
m2['transaction_size'] = m2['transaction_amt'].apply(
    lambda transaction_amt: getTransactionSize(transaction_amt))
m2['job_class'] = m2['occupation'].apply(
    lambda occupation: getJobClass(occupation))
m2['clusterName'] = m2['Cluster8Assignment'].apply(
    lambda cluster: renameClusters(cluster))
m2_priority = m2.merge(priority, left_on='Zipcode', right_on='Zipcode')
m2_cutoff = m2_priority[m2_priority['transaction_amt'] > 0]
m2_cutoff['logDollars'] = m2_cutoff['transaction_amt'].apply(lambda x: log(x))

# drop unused columns
df_clean = m2_cutoff.drop(['Unnamed: 0', 'transaction_amt', 'Cluster8Assignment',
                           'name', 'occupation', 'employer', 'ZIP_x', 'COUNTY', 'ZIP_y'], axis=1).dropna()

# Create a subset with n = 50,000 individual donations
# larger subsets take too long to load & render
df_clean_subset_50000 = df_clean.sample(n=50000)
df_clean_subset_50000.to_csv(
    '../04_data/df_clean_subset_50000.csv', header=True)
