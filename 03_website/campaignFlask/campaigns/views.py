# -*- coding: utf-8 -*-

import pandas as pd
from shapely.geometry import Point, shape

from flask import Flask
from flask import render_template
from flask import request
from campaigns import app
import json


data_path = './campaigns/input/'


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    
    df_clean = pd.read_csv(data_path + 'df_clean_subset_50000.csv', index_col = 0, converters={'Zipcode': lambda y: str(y)})

    return df_clean.to_json(orient='records')


#if __name__ == "__main__":
#    app.run(host='0.0.0.0',port=5000,debug=True)
