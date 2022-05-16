# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:04:50 2022

@author: sw5034
"""
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
import geopandas as gpd
#import IPython

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
spark

#Step1
PA_FN = '/tmp/bdm/weekly-patterns-nyc-2019-2020/*'
#PA_FN = '/home/sw5034/final/BDM_Final_Challenge/weekly-patterns-nyc-2019-2020-sample.csv'
pa = sc.textFile(PA_FN, use_unicode=True).cache()

def extractPA(partId, part):
  if partId == 0:
    next(part)
  import csv
  for record in csv.reader(part):
    placekey,startdate,enddate,poi_cbg,visitor_home_cbgs = record[0],record[12].split('T')[0],record[13].split('T')[0],record[18],record[19]
    yield placekey, ((startdate,enddate),poi_cbg,visitor_home_cbgs)


storeinfo = pa.mapPartitionsWithIndex(extractPA)

supermarket = pd.read_csv('nyc_supermarkets.csv')
supermarket = sc.parallelize([(supermarket['safegraph_placekey'].iloc[i],1) for i in range(len(supermarket))])

storeinfo2 = storeinfo.join(supermarket)\
          .mapValues(lambda x:x[0])

#Step2
def time_filter(time):
  year = time.split('-')[0]
  month = time.split('-')[1]
  if (year in ['2019','2020']) and (month in ['03','10']):
    return True

def uni_values(time):
  yearmonth = time[0].split('-')[0]+'-'+time[0].split('-')[1]
  if yearmonth.split('-')[1] == '02':
    return yearmonth.split('-')[0]+'-03'
  elif yearmonth.split('-')[1] == '09':
    return yearmonth.split('-')[0]+'-10'
  else:
    return yearmonth

storeinfo3 = storeinfo2.filter(lambda x: time_filter(x[1][0][0]) or time_filter(x[1][0][1]))\
            .mapValues(lambda x: (uni_values(x[0]),x[1],x[2]))
            
#Step3
nyccbg = pd.read_csv('nyc_cbg_centroids.csv')
geo_nyccbg = gpd.GeoDataFrame(nyccbg,geometry=gpd.points_from_xy(nyccbg['longitude'],nyccbg['latitude']),crs='EPSG:4326')
geo_nyccbg = geo_nyccbg.to_crs(crs='EPSG:2263')
nyccbg['lng_pro'] = geo_nyccbg['geometry'].x
nyccbg['lat_pro'] = geo_nyccbg['geometry'].y

nyccbg_rdd = sc.parallelize([(nyccbg['cbg_fips'].iloc[i],(nyccbg['lat_pro'].iloc[i], nyccbg['lng_pro'].iloc[i])) for i in range(len(nyccbg))])

def getmiles(meters):
  return meters*0.000621371192

nyccbg_rdd = nyccbg_rdd.map(lambda x: (str(x[0]),(getmiles(x[1][0]), getmiles(x[1][1]))))

storeinfo4 = storeinfo3.map(lambda x: (eval(x[1][2]),(x[0],x[1][0],x[1][1])))\
            .flatMap(lambda x: ((key,(x[1][0],x[1][2],x[1][1],value)) for (key,value) in x[0].items()))\
            .join(nyccbg_rdd)\
            .map(lambda x: (x[1][0][1],(x[1][0][2],x[0],x[1][1],x[1][0][3])))\
            .join(nyccbg_rdd)\
            .map(lambda x: (x[1][0][1],(x[1][0][0],x[1][0][2],x[1][1],x[1][0][3])))
            
#Step4
def distance(o_lat,o_lng,d_lat,d_lng):
  return ((o_lat-d_lat)**2+(o_lng-d_lng)**2)**0.5

storeinfo5 = storeinfo4.mapValues(lambda x: (x[0],distance(x[1][0],x[1][1],x[2][0],x[2][1]),x[3]))\
        .map(lambda x: ((x[0],x[1][0]),(x[1][1],x[1][2])))

#Step5
results = storeinfo5.reduceByKey(lambda x,y: (x[0]*x[1]+y[0]*y[1], x[1]+y[1]))\
  .mapValues(lambda x: round((x[0]/x[1]),2))\
  .map(lambda x: (x[0][0], (x[0][1], x[1])))\
  .mapValues(lambda x: [x[0],x[1]])\
  .reduceByKey(lambda x,y: x+y)  

def full_column(line,period_lst):
  distance_lst = []
  for period in period_lst:
    if period in line:
      dist = line[line.index(period)+1]
    else:
      dist = np.nan
    distance_lst.append(dist)
  return tuple(distance_lst)

results_final = results.mapValues(lambda x: full_column(x,['2019-03','2019-10','2020-03','2020-10']))\
                      .map(lambda x: (x[0],float(x[1][0]),float(x[1][1]),float(x[1][2]),float(x[1][3])))
                     
#Save the result
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, FloatType
spark = SparkSession(sc)
deptSchema = StructType([       
    StructField('cbg_fips', StringType(), True),
    StructField('2019-03', FloatType(), True),
    StructField('2019-10', FloatType(), True),
    StructField('2020-03', FloatType(), True),
    StructField('2020-10', FloatType(), True)
])
df = spark.createDataFrame(results_final, schema = deptSchema)
df = df.sort(df.cbg_fips)
df.write.option("header",True).csv(sys.argv[1] if len(sys.argv)>1 else 'result_base')


#spark-submit --executor-cores 5 --num-executors 5 --files nyc_supermarkets.csv,nyc_cbg_centroids.csv BDM_Final_sw5034.py result_base

