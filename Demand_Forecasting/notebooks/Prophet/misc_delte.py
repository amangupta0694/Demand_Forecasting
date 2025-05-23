# Databricks notebook source
df = spark.sql("SELECT * FROM `arc-dbx-uc`.demand_forecast.demand_forecast_train")
print(df.limit(5).toPandas())
print(df.count())