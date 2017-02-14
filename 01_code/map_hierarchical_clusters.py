# from hierarchically-clustered zip-codes to map
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
import folium
import json
import os
from branca.colormap import linear

%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (20, 12)
matplotlib.style.use('ggplot')


# load geoJSON zip code boundaries
zips = '../04_data/json/zips_us_geo.json'
geo_json_data = json.load(open(zips))

# load results from hierarchical clustering
zipclusters8 = pd.read_csv('../04_data/zipCodes_imputed_normalized_hierarchical_8Nodes_clusterAssignments.csv',
                           index_col=False, converters={'Zipcode': lambda y: str(y)})

colormap = linear.Set1.scale(
    zipclusters8.Cluster8Assignment.min(),
    zipclusters8.Cluster8Assignment.max())


# colormap
colormap = colormap.to_step(n=8)
zipcodes8_dict = zipclusters8.set_index('Zipcode')['Cluster8Assignment']
m = folium.Map([43, -100], zoom_start=4)

folium.GeoJson(
    geo_json_data,
    style_function=lambda feature: {
        'fillColor': colormap(zipcodes8_dict[str(feature['properties']['zip'])]) if
        str(feature['properties']['zip']) in zipcodes8_dict else '#ababad',
        'color': 'black',
        'weight': 0.05,
        'dashArray': '5, 5',
        'fillOpacity': 0.9,
    }
).add_to(m)

m.save('../04_data/zipCodes_map_8Nodes_clusterAssignments.html')
