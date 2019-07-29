# Bigdata_Bighair

This repository contains several analyses and samples from a hair dataset. Important features:

**Hair_Maps** contains the hair "heat maps" for every image in the yearbook dataset. The maps are probabilistic (scaled 0-1).

**Looks_by_Decades** contains a folder of sample images representing every decade from 1930-2010. An accompanying .csv file provides demographic info (gender, decade) about each picture.

**Size_by_Year** contains several analysis scripts to 
+ Measure hair density in every picture in the dataset, 
  - The script 1_Measure_Hair_Density.R puts out hair_f_density.csv and hair_m_density.csv, providing the hair density in all the male and all the female yearbook photos.
  
+ Analyze the median hair size from each year, and 
  - The script 2_Analyze_Size_Trends.R takes the .csv files from the first step and generates trend_f_smoothed.csv and trend_m_smoothed.csv, which give the median hair size from each year as well as the smoothed time series.
  
+ Estimate the hair size in the sample images in the "Looks_by_Decades" directory. 
  - The script 3_Size_in_Sample_Images.R estimates the hair density in every image selected as representative of a decade (in Looks_by_Decades) and outputs the estimates in Sample_image_hair_size.csv.
  
