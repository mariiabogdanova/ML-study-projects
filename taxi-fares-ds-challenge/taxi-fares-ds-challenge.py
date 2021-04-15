# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.ensemble import IsolationForest


#Loading the data
df = pd.read_csv('nyc-taxi-yellow-2015000000000000', sep=',', delimiter = ",") #insert your route

#Checking if loaded correctly
def basic_details(df):
    b = pd.DataFrame()
    b['Missing values'] = df.isnull().sum()
    b['N unique values'] = df.nunique()
    b['dtype'] = df.dtypes
    return b
print(basic_details(df))

#Plots before outlier removal 
sns.set(rc={'figure.figsize':(15, 7)})
sns.distplot(df['tip_amount'].values, axlabel = 'tip_amount', bins = 30)
plt.show()

sns.scatterplot(data=df, x="tip_amount", y = "total_amount")
plt.show()

#Outlier Detection
clf = IsolationForest(random_state = 42, contamination = 0.01)
df['Anomaly'] = clf.fit_predict(df[['total_amount', 'tip_amount']])

#Outlier visualisation
plt.title("Outlier vs. Normal Trips")
plt.rcParams['figure.figsize'] = [15, 7]
plt.scatter(df.loc[df.Anomaly == -1, ['total_amount']], df.loc[df.Anomaly == -1, ['tip_amount']], c='red')
plt.scatter(df.loc[df.Anomaly == 1, ['total_amount']], df.loc[df.Anomaly == 1, ['tip_amount']], c='green')
plt.show()

#Outlier removal
df_copy = df #taking a copy just in case
df = df.loc[df['Anomaly'] == 1].copy()
#Taking into account only tips and total payments higher than 0
df = df[df['tip_amount'] > 0] 
df = df[df['total_amount'] > 0]

#Plots after outlier removal 
sns.set(rc={'figure.figsize':(15, 7)})
sns.distplot(df['tip_amount'].values, axlabel = 'tip_amount', bins = 30)
plt.show()

sns.scatterplot(data=df, x="tip_amount", y = "total_amount")
plt.show()

#Dividing the data
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df = df.set_index('pickup_datetime')

night_df = df.between_time('21:00','05:00')
morning_df = df.between_time('05:00','11:00')
day_df = df.between_time('11:00','17:00')
evening_df = df.between_time('17:00','21:00')

df = df.reset_index()
df['weekday'] = df['pickup_datetime'].dt.dayofweek

vendor1_df = df[df['vendor_id'] == 1]
vendor2_df = df[df['vendor_id'] == 2]

weekend_df = df[((df['pickup_datetime'] > '2015-09-05 00:00:01') & (df['pickup_datetime'] < '2015-09-06 23:59:59')) | ((df['pickup_datetime'] > '2015-09-12 00:00:01') & (df['pickup_datetime'] < '2015-09-13 23:59:59')) | ((df['pickup_datetime'] > '2015-09-19 00:00:01') & (df['pickup_datetime'] < '2015-09-20 23:59:59')) | ((df['pickup_datetime'] > '2015-09-26 00:00:01') & (df['pickup_datetime'] < '2015-09-27 23:59:59'))]
weekend_df.head()
weekday_df = df[((df['pickup_datetime'] > '2015-09-01 00:00:01') & (df['pickup_datetime'] < '2015-09-04 23:59:59')) | ((df['pickup_datetime'] > '2015-09-07 00:00:01') & (df['pickup_datetime'] < '2015-09-11 23:59:59')) | ((df['pickup_datetime'] > '2015-09-14 00:00:01') & (df['pickup_datetime'] < '2015-09-18 23:59:59')) | ((df['pickup_datetime'] > '2015-09-21 00:00:01') & (df['pickup_datetime'] < '2015-09-25 23:59:59')) | ((df['pickup_datetime'] > '2015-09-28 00:00:01') & (df['pickup_datetime'] < '2015-09-30 23:59:59'))]

monday_df = df[df['weekday'] == 0]
tuesday_df = df[df['weekday'] == 1]
wednesday_df = df[df['weekday'] == 2]
thursday_df = df[df['weekday'] == 3]
friday_df = df[df['weekday'] == 4]
saturday_df = df[df['weekday'] == 5]
sunday_df = df[df['weekday'] == 6]

#Checking the results - UNCOMMENTED FOR SIMPLICITY 
# print('Vendor 1')
# print(vendor1_df['tip_amount'].describe().round(3))
# print(' ')
# print('Vendor 2')
# print(vendor2_df['tip_amount'].describe().round(3))
# print('________________________________________________')
# print(' ')
# print('Weekdays')
# print(weekday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Weekends')
# print(weekend_df['tip_amount'].describe().round(3))
# print('________________________________________________')
# print(' ')
# print('Night')
# print(weekday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Morning')
# print(weekend_df['tip_amount'].describe().round(3))
# print(' ')
# print('Daytime')
# print(weekday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Evening')
# print(weekend_df['tip_amount'].describe().round(3))
# print('________________________________________________')
# print(' ')
# print('Monday')
# print(monday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Tuesday')
# print(tuesday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Wednesday')
# print(wednesday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Thursday')
# print(thursday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Friday')
# print(friday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Saturday')
# print(saturday_df['tip_amount'].describe().round(3))
# print(' ')
# print('Sunday')
# print(sunday_df['tip_amount'].describe().round(3))
# print('________________________________________________')
# print(' ')
# print('Monday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 0]['tip_amount'].describe())
# print(' ')
# print('Tuesday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 1]['tip_amount'].describe())
# print(' ')
# print('Wednesday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 2]['tip_amount'].describe())
# print(' ')
# print('Thursday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 3]['tip_amount'].describe())
# print(' ')
# print('Friday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 4]['tip_amount'].describe())
# print(' ')
# print('Saturday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 5]['tip_amount'].describe())
# print(' ')
# print('Sunday - vendor 1')
# print(vendor1_df[vendor1_df['weekday'] == 6]['tip_amount'].describe())
# print('________________________________________________')
# print(' ')
# print('Monday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 0]['tip_amount'].describe())
# print(' ')
# print('Tuesday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 1]['tip_amount'].describe())
# print(' ')
# print('Wednesday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 2]['tip_amount'].describe())
# print(' ')
# print('Thursday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 3]['tip_amount'].describe())
# print(' ')
# print('Friday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 4]['tip_amount'].describe())
# print(' ')
# print('Saturday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 5]['tip_amount'].describe())
# print(' ')
# print('Sunday - vendor 2')
# print(vendor2_df[vendor2_df['weekday'] == 6]['tip_amount'].describe())
# print('________________________________________________')
# print(' ')

# vendor1_df = vendor1_df.set_index('pickup_datetime')
# vendor2_df = vendor2_df.set_index('pickup_datetime')

# print('Night - vendor 1')
# print(vendor1_df.between_time('21:00','05:00')['tip_amount'].describe())
# print(' ')
# print('Morning - vendor 1')
# print(vendor1_df.between_time('05:00','11:00')['tip_amount'].describe())
# print(' ')
# print('Daytime - vendor 1')
# print(vendor1_df.between_time('11:00','17:00')['tip_amount'].describe())
# print(' ')
# print('Evening - vendor 1')
# print(vendor1_df.between_time('17:00','21:00')['tip_amount'].describe())
# print(' ')
# print('________________________________________________')
# print('Night - vendor 2')
# print(vendor2_df.between_time('21:00','05:00')['tip_amount'].describe())
# print(' ')
# print('Morning - vendor 2')
# print(vendor2_df.between_time('05:00','11:00')['tip_amount'].describe())
# print(' ')
# print('Daytime - vendor 2')
# print(vendor2_df.between_time('11:00','17:00')['tip_amount'].describe())
# print(' ')
# print('Evening - vendor 2')
# print(vendor2_df.between_time('17:00','21:00')['tip_amount'].describe())

#T-tests
print("Vendors comparison")
print("The probability of the accidental difference is higher that 5%: ", stats.ttest_ind(np.array(vendor1_df['tip_amount'].values), np.array(vendor2_df['tip_amount'].values)).pvalue > 0.05) #
print("T-test results: ", stats.ttest_ind(np.array(vendor1_df['tip_amount'].values), np.array(vendor2_df['tip_amount'].values)))
print(' ')
print('Weekend VS weekday comparison')
print("The probability of the accidental difference is higher that 5%: ", stats.ttest_ind(np.array(weekend_df['tip_amount'].values), np.array(weekday_df['tip_amount'].values)).pvalue > 0.05) #
print("T-test results: ", stats.ttest_ind(np.array(weekend_df['tip_amount'].values), np.array(weekday_df['tip_amount'].values)))
print(' ')
print('Monday vs Saturday')
print("The probability of the accidental difference is higher that 5%: ", stats.ttest_ind(np.array(monday_df['tip_amount'].values), np.array(saturday_df['tip_amount'].values)).pvalue > 0.05) #
print("T-test results: ", stats.ttest_ind(np.array(monday_df['tip_amount'].values), np.array(sunday_df['tip_amount'].values)))
print(' ')
print('Morning vs Evening')
print("The probability of the accidental difference is higher that 5%: ", stats.ttest_ind(np.array(day_df['tip_amount'].values), np.array(evening_df['tip_amount'].values)).pvalue > 0.05) #
print("T-test results: ", stats.ttest_ind(np.array(day_df['tip_amount'].values), np.array(evening_df['tip_amount'].values)))

#Boxplots of the data
sns.boxplot(x="weekday", y="tip_amount", data=df, showfliers = False)
plt.show()

sns.boxplot(x="rate_code", y="tip_amount", data=df, showfliers = False)
plt.show()

sns.boxplot(x="vendor_id", y="tip_amount", data=df, showfliers = False)
plt.show()