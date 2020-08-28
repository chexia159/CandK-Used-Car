# ==================================
# @Time        :Aug 17 2020
# @Author      :Da Ban
# @FileName    :DM_Test.py
# @Software    :Visual Studio Code
# ==================================

#%matplotlib inline
#%pylab inline



# %%
import os
import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams
import datetime as dt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt

from collections import Counter
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
# %matplotlib inline


# %%
path_k = './Encar_Data.xlsx'
rdfk = pd.read_excel(path_k, sheet_name=None)
dfk = rdfk['K_Used_Car']
##print(dfk)

# %%
dfk = dfk.drop(dfk[dfk.Transmission == 'MT'].index)
dfk = dfk.drop(dfk[dfk.Turbo > 0].index)
dfk = dfk.drop(dfk[(dfk.PriceInDollars > 50000) | (dfk.PriceInDollars < 5000)].index)
dfk = dfk.drop(dfk[dfk.Mileage > 400000].index)

# %%
path_c = './168_Data.xlsx'
rdfc = pd.read_excel(path_c, sheet_name=None)
##print(rdfc)
dfc = rdfc['168_Used_Car']
##print(dfc)

# %%
dfc = dfc.drop(dfc[dfc.Transmission == 'MT'].index)
dfc = dfc.drop(dfc[(dfc.Turbo_Energy == 1) | (dfc.Turbo_Elite == 1) | (dfc.Turbo_Premium == 1)].index)
dfc = dfc.drop(dfc[dfc.Age < 1.5].index)
milecheck = dfc['Mileage'].iloc[0]
if milecheck < 50:
    dfc['Mileage']=dfc['Mileage']*10000
dfc = dfc.dropna()

# %%
dfk.isnull().any()

# %%
dfc.isnull().any()

# %%
dfk.dtypes

# %%
dfc.dtypes

# %%
dfk.describe()

# %%
dfc.describe()

# %%
df_price = dfk.filter(['PriceInDollars'],axis = 1)
df_price = df_price.rename(columns={'PriceInDollars':'K_Price'})
df_price['C_Price'] = dfc.filter(['PriceInDollars'],axis = 1) 
##plt.figure(figsize=(7,4))
sns.set(style='whitegrid')
ax1 = sns.boxplot(data=df_price,width=0.3)
ax1 = sns.swarmplot(data=df_price,size=1,color='.2',linewidth = 0)
ax1.set(ylabel='Price')

# %%
df_age = dfk.filter(['Age'],axis = 1)
df_age = df_age.rename(columns={'Age':'K_Age'})
df_age['C_Age'] = dfc.filter(['Age'],axis = 1) 
sns.set(style='whitegrid')
ax2 = sns.boxplot(data=df_age,width=0.3)
ax2 = sns.swarmplot(data=df_age,size=1,color='.2',linewidth = 0)
ax2.set(ylabel='Age')

# %%
df_mileage = dfk.filter(['Mileage'],axis = 1)
df_mileage = df_mileage.rename(columns={'Mileage':'K_Mileage'})
df_mileage['C_Mileage'] = dfc.filter(['Mileage'],axis = 1) 
sns.set(style='whitegrid')
ax2 = sns.boxplot(data=df_mileage,width=0.3)
ax2 = sns.swarmplot(data=df_mileage,size=1,color='.2',linewidth = 0)
ax2.set(ylabel='Mileage(km)')

# %%
ax3 = sns.distplot(df_price[['K_Price']],label='Korea',hist=False,rug=True)
ax3 = sns.distplot(df_price[['C_Price']],label='China',hist=False,rug=True)
ax3.set(xlabel='Price')
ax3 = plt.show


# %%
ax4 = sns.distplot(df_age[['K_Age']],label='Korea',hist=False,rug=True)
ax4 = sns.distplot(df_age[['C_Age']],label='China',hist=False,rug=True)
ax4.set(xlabel='Age')
ax4 = plt.show

# %%
ax5 = sns.distplot(df_mileage[['K_Mileage']],label='Korea',hist=False,rug=True)
ax5 = sns.distplot(df_mileage[['C_Mileage']],label='China',hist=False,rug=True)
ax5.set(xlabel='Mileage(km)')
ax5 = plt.show

# %%
ax6 = sns.distplot(df_mileage[['C_Mileage']],label='China',hist=True)
ax6.set(xlabel='Mileage(km)')
ax6 = plt.show

# %% 
ax7 = sns.distplot(df_mileage[['K_Mileage']],label='Korea',hist=True)
ax7.set(xlabel='Mileage(km)')
ax7 = plt.show

# %%
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
stats.probplot(dfc['Mileage'],plot=sns.mpl.pyplot)
plt.xlabel('C_Mileage')
plt.subplot(1,2,2)
stats.probplot(dfk['Mileage'],plot=sns.mpl.pyplot)
plt.xlabel('K_Mileage')

# %%
g1 = sns.PairGrid(dfk, vars=['PriceInDollars', 'Age', 'Mileage'], hue='Trim')
g1.map_diag(plt.hist)
g1.map_offdiag(plt.scatter)
g1.add_legend()

# %%
g2 = sns.PairGrid(dfc, vars=['PriceInDollars', 'Age', 'Mileage'], hue='Trim')
g2.map_diag(plt.hist)
g2.map_offdiag(plt.scatter)
g2.add_legend()

# %%
yk = dfk['PriceInDollars']
yc = dfc['PriceInDollars']
xk = dfk.drop(['Trim','Price','Transmission','PriceInDollars','Turbo'], axis = 1)
xc = dfc.drop(['Trim','Price','Transmission','PriceInDollars','Turbo_Energy','Turbo_Elite','Turbo_Premium'], axis = 1)

# %%
xk_train, xk_test, yk_train, yk_test = train_test_split(xk, yk)
xc_train, xc_test, yc_train, yc_test = train_test_split(xc, yc)

# %%
rmse_val = []
for k in range(14):
    k = k+1 
    model = neighbors.KNeighborsRegressor(n_neighbors= k )
    model.fit(xk_train,yk_train)
    pred = model.predict(xk_test)
    error = sqrt(mean_squared_error(yk_test,pred))
    rmse_val.append(error)
    print('RMSE value for k =' , k, 'is:', error)
curve = pd.DataFrame(rmse_val, columns=['KNN'])


# %%
rmse_val2 = []
for k in range(10):
    k = k+1 
    model = neighbors.KNeighborsRegressor(n_neighbors= k )
    model.fit(xc_train,yc_train)
    pred = model.predict(xc_test)
    error = sqrt(mean_squared_error(yc_test,pred))
    rmse_val2.append(error)
    print('RMSE value for k =' , k, 'is:', error)
curve2 = pd.DataFrame(rmse_val2, columns=['KNN'])

# %%
sns.set(style='darkgrid')
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.lineplot(data=curve)
plt.xlabel('Korean K Value')
plt.ylabel('RMSE')
plt.subplot(1,2,2)
sns.lineplot(data=curve2)
plt.xlabel('Chinese K Value')
plt.ylabel('RMSE')

# %%   
yk = dfk['PriceInDollars']
yc = dfc['PriceInDollars']
xk = dfk.filter(['Mileage'],axis = 1)
xc = dfc.filter(['Mileage'],axis = 1)

# %%
xk_train, xk_test, yk_train, yk_test = train_test_split(xk, yk)
xc_train, xc_test, yc_train, yc_test = train_test_split(xc, yc)
rmse1 = []
rmse2 = []
degrees = np.arange(1,10)

# %%
for deg in degrees:
    poly_features = PolynomialFeatures(degree = deg,include_bias=False)
    xk_poly_train = poly_features.fit_transform(xk_train)
    
    poly_reg = LinearRegression()
    poly_reg.fit(xk_poly_train,yk_train)

    xk_poly_test = poly_features.fit_transform(xk_test)
    poly_predict = poly_reg.predict(xk_poly_test)
    rmse = sqrt(mean_squared_error(yk_test,poly_predict))
    rmse1.append(rmse)


# %%
for deg in degrees:
    poly_features = PolynomialFeatures(degree = deg,include_bias=False)
    xc_poly_train = poly_features.fit_transform(xc_train)
    
    poly_reg = LinearRegression()
    poly_reg.fit(xc_poly_train,yc_train)

    xc_poly_test = poly_features.fit_transform(xc_test)
    poly_predict = poly_reg.predict(xc_poly_test)
    rmse = sqrt(mean_squared_error(yc_test,poly_predict))
    rmse2.append(rmse)

# %%
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.lineplot(x = degrees, y = rmse1)
plt.xlabel('K_Degrees')
plt.ylabel('RMSE')
plt.subplot(1,2,2)
sns.lineplot(x = degrees, y = rmse2)
plt.xlabel('C_Degrees')
plt.ylabel('RMSE')
plt.suptitle('Polinomial Validation In Terms Of Mileage')

# %%
yk = dfk['PriceInDollars']
yc = dfc['PriceInDollars']
xk = dfk.filter(['Age'],axis = 1)
xc = dfc.filter(['Age'],axis = 1)

xk_train, xk_test, yk_train, yk_test = train_test_split(xk, yk)
xc_train, xc_test, yc_train, yc_test = train_test_split(xc, yc)
rmse1 = []
rmse2 = []
degrees = np.arange(1,10)

for deg in degrees:
    poly_features = PolynomialFeatures(degree = deg,include_bias=False)
    xk_poly_train = poly_features.fit_transform(xk_train)
    
    poly_reg = LinearRegression()
    poly_reg.fit(xk_poly_train,yk_train)

    xk_poly_test = poly_features.fit_transform(xk_test)
    poly_predict = poly_reg.predict(xk_poly_test)
    rmse = sqrt(mean_squared_error(yk_test,poly_predict))
    rmse1.append(rmse)

for deg in degrees:
    poly_features = PolynomialFeatures(degree = deg,include_bias=False)
    xc_poly_train = poly_features.fit_transform(xc_train)
    
    poly_reg = LinearRegression()
    poly_reg.fit(xc_poly_train,yc_train)

    xc_poly_test = poly_features.fit_transform(xc_test)
    poly_predict = poly_reg.predict(xc_poly_test)
    rmse = sqrt(mean_squared_error(yc_test,poly_predict))
    rmse2.append(rmse)

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.lineplot(x = degrees, y = rmse1)
plt.xlabel('K_Degrees')
plt.ylabel('RMSE')
plt.subplot(1,2,2)
sns.lineplot(x = degrees, y = rmse2)
plt.xlabel('C_Degrees')
plt.ylabel('RMSE')
plt.suptitle('Polinomial Validation In Terms Of Age')


# %%
dfk_train, dfk_test = train_test_split(dfk)
model1 = ols(formula = 'PriceInDollars ~ Mileage + Age + Inspection + Record + Compensate + Style + Smart + Modern + Value_Plus + Premium', data = dfk).fit()
print(model1.summary())

# %%
#model1 = ols(formula = 'PriceInDollars ~ Mileage + Age + Inspection + Record + Compensate + Style + Smart + Modern + Value_Plus + Premium', data = dfk_train).fit()
#k_predict = model1.predict(dfk_test)
#rmse1 = sqrt(mean_squared_error(dfk_test['PriceInDollars'],k_predict))
#print('RMSE for the model is', rmse1)

# %%
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

# %%
X = dfk.iloc[:,2:-2]
X = X.drop(['Transmission','Turbo'], axis = 1)
calc_vif(X)

# %%
sns.scatterplot(model1.model.exog[:,1],model1.resid)
sns.lineplot(model1.model.exog[:,1],y = 0, color = 'red')

# %%
fitted_value = model1.fittedvalues
sns.residplot(fitted_value, dfk.columns[-1], data=dfk,lowess=True,line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.title('Residuals')

# %%
plt.figure(figsize=(10,6))
_, upper, lower = wls_prediction_std(model1)
ypred = model1.predict(dfk)
sns.scatterplot(x=dfk['Mileage'],y=dfk['PriceInDollars'])
sns.lineplot(x=dfk['Mileage'],y=ypred)

# %%
dfc_train, dfc_test = train_test_split(dfc)
model2 = ols(formula = 'PriceInDollars ~ Age + Record + Youth + Elite + Anniv + Luxury + Premium', data = dfc).fit()
print(model2.summary())

# %%
X = dfc.iloc[:,2:-2]
X = X.drop(['Mileage','Transmission','Turbo_Energy','Turbo_Elite','Turbo_Premium','Energy' ], axis = 1)
calc_vif(X)

# %%
model2 = ols(formula = 'PriceInDollars ~ Age + Record + Youth + Elite + Anniv + Luxury + Premium', data = dfc_train).fit()
c_predict = model2.predict(dfc_test)
rmse2 = sqrt(mean_squared_error(dfc_test['PriceInDollars'],c_predict))
print('RMSE for the model is', rmse2)

# %%
dfk_elite = pd.DataFrame
dfk_elite = dfk[dfk['Value_Plus'] == 1]
dfk_elite = dfk_elite[['Age','Mileage','Record','PriceInDollars']]
df_elite = pd.DataFrame
df_elite = dfk_elite.copy()
df_elite['C_Intcpt'] = 0
df_elite['C_Mlg'] = 0 
df_elite['C_Age'] = 0
df_elite['Country'] = 'Korea'
dfc_elite = dfc[dfc['Elite'] == 1]
dfc_elite = dfc_elite[['Age','Mileage','Record','PriceInDollars']]
dfc_elite['C_Intcpt'] = 1
dfc_elite['C_Mlg'] = 1
dfc_elite['C_Age'] = 1
dfc_elite['Country'] = 'China'
df_elite = pd.concat([df_elite,dfc_elite], ignore_index=True)
df_elite['C_Mlg'] = df_elite['C_Mlg'] * df_elite['Mileage']
df_elite['C_Age'] = df_elite['C_Age'] * df_elite['Age']
sns.scatterplot(x='Age', y='PriceInDollars', hue='Country',palette='ch:r=-.2,d=.3_r',data=df_elite)

# %%
model_elite_k = ols(formula='PriceInDollars ~ Age + Mileage + Record', data = dfk_elite).fit()
print(model_elite_k.summary())

# %%
dfc_elite = dfc_elite.drop(dfc_elite[dfc_elite.Age < 1].index)
model_elite_c = ols(formula='PriceInDollars ~ Age + Mileage + Record', data = dfc_elite).fit()
print(model_elite_c.summary())

# %%
df_elite = df_elite.drop(df_elite[df_elite.Age < 1].index)
model_elite = ols(formula='PriceInDollars ~ Age + Mileage + Record + C_Intcpt + C_Age + C_Mlg', data = df_elite).fit()
print(model_elite.summary())

# %%
df_elite2 = df_elite.drop(df_elite[df_elite.Record == 0].index)
df_elite2 = df_elite2.drop(df_elite2[df_elite2.Age < 1.5].index)
model_elite_2 = ols(formula='PriceInDollars ~ Age + C_Intcpt + C_Age', data = df_elite2).fit()
print(model_elite_2.summary())

# %%
g = sns.lmplot(x='Age',y='PriceInDollars',hue = 'Country',x_ci='sd', data=df_elite2)
g.set(xlim=(1,5))
# %%
dfk_lowspec = pd.DataFrame
dfk_lowspec = dfk[dfk['Style'] == 1]
dfk_lowspec = dfk_lowspec[['Age','PriceInDollars']]

# %%
model_lowspec_k = ols(formula='PriceInDollars ~ Age', data=dfk_lowspec).fit()
print(model_lowspec_k.summary())
model_elite_k2 = ols(formula='PriceInDollars ~ Age', data=dfk_elite).fit()
print(model_elite_k2.summary())
model_lowspec_c = ols(formula='PriceInDollars ~ Age', data=dfc_elite).fit()
print(model_lowspec_c.summary())

# %%
pred_df = pd.DataFrame({'Age':[2]})
pred_lowspec_k = model_lowspec_k.predict(pred_df)
print('The predict price of a 2-year Style(K) trim is', pred_lowspec_k[0],'Dollars')
pred_lowspec_k2 = model_elite_k2.predict(pred_df)
print('The predict price of a 2-year Value Plus(K) trim is', pred_lowspec_k2[0],'Dollars')
pred_lowspec_c = model_lowspec_c.predict(pred_df)
print('The predict price of a 2-year Elite(C) trim is', pred_lowspec_c[0],'Dollars')


# %%



# %%
