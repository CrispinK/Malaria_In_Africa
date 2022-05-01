# MALARIA IN AFRICA EDA


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
```


```python
Malaria = pd.read_csv('Malaria.csv')
Malaria.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>Year</th>
      <th>Country Code</th>
      <th>Incidence of malaria (per 1,000 population at risk)</th>
      <th>Malaria cases reported</th>
      <th>Use of insecticide-treated bed nets (% of under-5 population)</th>
      <th>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)</th>
      <th>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)</th>
      <th>People using safely managed drinking water services (% of population)</th>
      <th>People using safely managed drinking water services, rural (% of rural population)</th>
      <th>...</th>
      <th>Urban population growth (annual %)</th>
      <th>People using at least basic drinking water services (% of population)</th>
      <th>People using at least basic drinking water services, rural (% of rural population)</th>
      <th>People using at least basic drinking water services, urban (% of urban population)</th>
      <th>People using at least basic sanitation services (% of population)</th>
      <th>People using at least basic sanitation services, rural (% of rural population)</th>
      <th>People using at least basic sanitation services, urban  (% of urban population)</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Algeria</td>
      <td>2007</td>
      <td>DZA</td>
      <td>0.01</td>
      <td>26.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.71</td>
      <td>91.68</td>
      <td>85.83</td>
      <td>94.78</td>
      <td>85.85</td>
      <td>76.94</td>
      <td>90.57</td>
      <td>28.033886</td>
      <td>1.659626</td>
      <td>POINT (28.033886 1.659626)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Angola</td>
      <td>2007</td>
      <td>AGO</td>
      <td>286.72</td>
      <td>1533485.0</td>
      <td>18.0</td>
      <td>29.8</td>
      <td>1.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.01</td>
      <td>47.96</td>
      <td>23.77</td>
      <td>65.83</td>
      <td>37.26</td>
      <td>14.00</td>
      <td>54.44</td>
      <td>-11.202692</td>
      <td>17.873887</td>
      <td>POINT (-11.202692 17.873887)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Benin</td>
      <td>2007</td>
      <td>BEN</td>
      <td>480.24</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.09</td>
      <td>63.78</td>
      <td>54.92</td>
      <td>76.24</td>
      <td>11.80</td>
      <td>4.29</td>
      <td>22.36</td>
      <td>9.307690</td>
      <td>2.315834</td>
      <td>POINT (9.307689999999999 2.315834)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Botswana</td>
      <td>2007</td>
      <td>BWA</td>
      <td>1.03</td>
      <td>390.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.80</td>
      <td>78.89</td>
      <td>57.60</td>
      <td>94.35</td>
      <td>61.60</td>
      <td>39.99</td>
      <td>77.30</td>
      <td>-22.328474</td>
      <td>24.684866</td>
      <td>POINT (-22.328474 24.684866)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Burkina Faso</td>
      <td>2007</td>
      <td>BFA</td>
      <td>503.80</td>
      <td>44246.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>5.91</td>
      <td>52.27</td>
      <td>45.13</td>
      <td>76.15</td>
      <td>15.60</td>
      <td>6.38</td>
      <td>46.49</td>
      <td>12.238333</td>
      <td>-1.561593</td>
      <td>POINT (12.238333 -1.561593)</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
#shape of data
Malaria.shape
```




    (594, 27)




```python
Malaria.dtypes
```




    Country Name                                                                                object
    Year                                                                                         int64
    Country Code                                                                                object
    Incidence of malaria (per 1,000 population at risk)                                        float64
    Malaria cases reported                                                                     float64
    Use of insecticide-treated bed nets (% of under-5 population)                              float64
    Children with fever receiving antimalarial drugs (% of children under age 5 with fever)    float64
    Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)      float64
    People using safely managed drinking water services (% of population)                      float64
    People using safely managed drinking water services, rural (% of rural population)         float64
    People using safely managed drinking water services, urban (% of urban population)         float64
    People using safely managed sanitation services (% of population)                          float64
    People using safely managed sanitation services, rural (% of rural population)             float64
    People using safely managed sanitation services, urban  (% of urban population)            float64
    Rural population (% of total population)                                                   float64
    Rural population growth (annual %)                                                         float64
    Urban population (% of total population)                                                   float64
    Urban population growth (annual %)                                                         float64
    People using at least basic drinking water services (% of population)                      float64
    People using at least basic drinking water services, rural (% of rural population)         float64
    People using at least basic drinking water services, urban (% of urban population)         float64
    People using at least basic sanitation services (% of population)                          float64
    People using at least basic sanitation services, rural (% of rural population)             float64
    People using at least basic sanitation services, urban  (% of urban population)            float64
    latitude                                                                                   float64
    longitude                                                                                  float64
    geometry                                                                                    object
    dtype: object




```python
#convert year to date time 
from datetime import datetime

Malaria['Year'] = pd.to_datetime(Malaria.Year,format='%Y')

Malaria.dtypes
```




    Country Name                                                                                       object
    Year                                                                                       datetime64[ns]
    Country Code                                                                                       object
    Incidence of malaria (per 1,000 population at risk)                                               float64
    Malaria cases reported                                                                            float64
    Use of insecticide-treated bed nets (% of under-5 population)                                     float64
    Children with fever receiving antimalarial drugs (% of children under age 5 with fever)           float64
    Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)             float64
    People using safely managed drinking water services (% of population)                             float64
    People using safely managed drinking water services, rural (% of rural population)                float64
    People using safely managed drinking water services, urban (% of urban population)                float64
    People using safely managed sanitation services (% of population)                                 float64
    People using safely managed sanitation services, rural (% of rural population)                    float64
    People using safely managed sanitation services, urban  (% of urban population)                   float64
    Rural population (% of total population)                                                          float64
    Rural population growth (annual %)                                                                float64
    Urban population (% of total population)                                                          float64
    Urban population growth (annual %)                                                                float64
    People using at least basic drinking water services (% of population)                             float64
    People using at least basic drinking water services, rural (% of rural population)                float64
    People using at least basic drinking water services, urban (% of urban population)                float64
    People using at least basic sanitation services (% of population)                                 float64
    People using at least basic sanitation services, rural (% of rural population)                    float64
    People using at least basic sanitation services, urban  (% of urban population)                   float64
    latitude                                                                                          float64
    longitude                                                                                         float64
    geometry                                                                                           object
    dtype: object




```python
#import regular packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
#import the plotly packages
import plotly.offline as py
from plotly import tools
import plotly.graph_objs as go
import plotly.express as px
```


```python
#Malaria data for plotting
Malaria_data = Malaria[['Country Name','Year','Country Code','Incidence of malaria (per 1,000 population at risk)','Malaria cases reported','Use of insecticide-treated bed nets (% of under-5 population)','Children with fever receiving antimalarial drugs (% of children under age 5 with fever)','Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)','latitude','longitude','geometry']]
```


```python
Malaria_data.dtypes
```




    Country Name                                                                                       object
    Year                                                                                       datetime64[ns]
    Country Code                                                                                       object
    Incidence of malaria (per 1,000 population at risk)                                               float64
    Malaria cases reported                                                                            float64
    Use of insecticide-treated bed nets (% of under-5 population)                                     float64
    Children with fever receiving antimalarial drugs (% of children under age 5 with fever)           float64
    Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)             float64
    latitude                                                                                          float64
    longitude                                                                                         float64
    geometry                                                                                           object
    dtype: object




```python
#convert year to string
Malaria_data.Year = Malaria_data.Year.astype(str)
```

    C:\Users\crispin\AppData\Local\Temp\ipykernel_13924\98639886.py:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    


 


```python
#some more data exploreation
Malaria.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 594 entries, 0 to 593
    Data columns (total 27 columns):
     #   Column                                                                                   Non-Null Count  Dtype         
    ---  ------                                                                                   --------------  -----         
     0   Country Name                                                                             594 non-null    object        
     1   Year                                                                                     594 non-null    datetime64[ns]
     2   Country Code                                                                             594 non-null    object        
     3   Incidence of malaria (per 1,000 population at risk)                                      550 non-null    float64       
     4   Malaria cases reported                                                                   550 non-null    float64       
     5   Use of insecticide-treated bed nets (% of under-5 population)                            132 non-null    float64       
     6   Children with fever receiving antimalarial drugs (% of children under age 5 with fever)  122 non-null    float64       
     7   Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)    106 non-null    float64       
     8   People using safely managed drinking water services (% of population)                    99 non-null     float64       
     9   People using safely managed drinking water services, rural (% of rural population)       88 non-null     float64       
     10  People using safely managed drinking water services, urban (% of urban population)       176 non-null    float64       
     11  People using safely managed sanitation services (% of population)                        132 non-null    float64       
     12  People using safely managed sanitation services, rural (% of rural population)           110 non-null    float64       
     13  People using safely managed sanitation services, urban  (% of urban population)          132 non-null    float64       
     14  Rural population (% of total population)                                                 588 non-null    float64       
     15  Rural population growth (annual %)                                                       588 non-null    float64       
     16  Urban population (% of total population)                                                 588 non-null    float64       
     17  Urban population growth (annual %)                                                       588 non-null    float64       
     18  People using at least basic drinking water services (% of population)                    588 non-null    float64       
     19  People using at least basic drinking water services, rural (% of rural population)       566 non-null    float64       
     20  People using at least basic drinking water services, urban (% of urban population)       566 non-null    float64       
     21  People using at least basic sanitation services (% of population)                        588 non-null    float64       
     22  People using at least basic sanitation services, rural (% of rural population)           566 non-null    float64       
     23  People using at least basic sanitation services, urban  (% of urban population)          566 non-null    float64       
     24  latitude                                                                                 594 non-null    float64       
     25  longitude                                                                                594 non-null    float64       
     26  geometry                                                                                 594 non-null    object        
    dtypes: datetime64[ns](1), float64(23), object(3)
    memory usage: 125.4+ KB
    


```python
Malaria.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Incidence of malaria (per 1,000 population at risk)</th>
      <th>Malaria cases reported</th>
      <th>Use of insecticide-treated bed nets (% of under-5 population)</th>
      <th>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)</th>
      <th>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)</th>
      <th>People using safely managed drinking water services (% of population)</th>
      <th>People using safely managed drinking water services, rural (% of rural population)</th>
      <th>People using safely managed drinking water services, urban (% of urban population)</th>
      <th>People using safely managed sanitation services (% of population)</th>
      <th>People using safely managed sanitation services, rural (% of rural population)</th>
      <th>...</th>
      <th>Urban population (% of total population)</th>
      <th>Urban population growth (annual %)</th>
      <th>People using at least basic drinking water services (% of population)</th>
      <th>People using at least basic drinking water services, rural (% of rural population)</th>
      <th>People using at least basic drinking water services, urban (% of urban population)</th>
      <th>People using at least basic sanitation services (% of population)</th>
      <th>People using at least basic sanitation services, rural (% of rural population)</th>
      <th>People using at least basic sanitation services, urban  (% of urban population)</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>550.000000</td>
      <td>5.500000e+02</td>
      <td>132.000000</td>
      <td>122.000000</td>
      <td>106.000000</td>
      <td>99.000000</td>
      <td>88.000000</td>
      <td>176.000000</td>
      <td>132.000000</td>
      <td>110.000000</td>
      <td>...</td>
      <td>588.000000</td>
      <td>588.000000</td>
      <td>588.000000</td>
      <td>566.000000</td>
      <td>566.000000</td>
      <td>588.000000</td>
      <td>566.000000</td>
      <td>566.000000</td>
      <td>594.000000</td>
      <td>594.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>190.087491</td>
      <td>1.068330e+06</td>
      <td>42.530303</td>
      <td>30.201639</td>
      <td>15.013958</td>
      <td>33.478990</td>
      <td>12.470568</td>
      <td>51.549545</td>
      <td>28.768939</td>
      <td>14.361727</td>
      <td>...</td>
      <td>43.164116</td>
      <td>3.523061</td>
      <td>65.994915</td>
      <td>51.449576</td>
      <td>84.268498</td>
      <td>39.469796</td>
      <td>28.077208</td>
      <td>48.088375</td>
      <td>2.828796</td>
      <td>17.342546</td>
    </tr>
    <tr>
      <th>std</th>
      <td>163.054527</td>
      <td>2.192802e+06</td>
      <td>20.157059</td>
      <td>18.903198</td>
      <td>12.389166</td>
      <td>26.678321</td>
      <td>10.078371</td>
      <td>24.157416</td>
      <td>18.631510</td>
      <td>7.088038</td>
      <td>...</td>
      <td>18.086118</td>
      <td>1.456244</td>
      <td>17.283361</td>
      <td>18.927868</td>
      <td>9.307285</td>
      <td>26.304934</td>
      <td>24.046725</td>
      <td>21.802128</td>
      <td>15.678226</td>
      <td>20.041257</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>5.770000</td>
      <td>0.930000</td>
      <td>11.200000</td>
      <td>6.370000</td>
      <td>2.300000</td>
      <td>...</td>
      <td>9.860000</td>
      <td>-4.650000</td>
      <td>28.960000</td>
      <td>17.050000</td>
      <td>52.010000</td>
      <td>4.990000</td>
      <td>1.890000</td>
      <td>12.580000</td>
      <td>-30.559482</td>
      <td>-24.013197</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.857500</td>
      <td>2.211750e+03</td>
      <td>26.675000</td>
      <td>17.275000</td>
      <td>5.763285</td>
      <td>8.975000</td>
      <td>4.185000</td>
      <td>34.125000</td>
      <td>16.532500</td>
      <td>7.200000</td>
      <td>...</td>
      <td>28.795000</td>
      <td>2.512500</td>
      <td>52.375000</td>
      <td>37.075000</td>
      <td>78.080000</td>
      <td>18.197500</td>
      <td>8.842500</td>
      <td>30.775000</td>
      <td>-6.369028</td>
      <td>0.824782</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>174.775000</td>
      <td>1.130260e+05</td>
      <td>42.900000</td>
      <td>29.300000</td>
      <td>11.500000</td>
      <td>28.390000</td>
      <td>10.675000</td>
      <td>51.365000</td>
      <td>25.410000</td>
      <td>15.950000</td>
      <td>...</td>
      <td>41.560000</td>
      <td>3.730000</td>
      <td>64.470000</td>
      <td>50.435000</td>
      <td>85.420000</td>
      <td>32.555000</td>
      <td>18.815000</td>
      <td>44.695000</td>
      <td>6.744051</td>
      <td>18.611308</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>347.637500</td>
      <td>1.154808e+06</td>
      <td>56.325000</td>
      <td>42.625000</td>
      <td>21.850000</td>
      <td>43.890000</td>
      <td>16.887500</td>
      <td>70.747500</td>
      <td>35.725000</td>
      <td>20.315000</td>
      <td>...</td>
      <td>56.945000</td>
      <td>4.450000</td>
      <td>79.165000</td>
      <td>62.245000</td>
      <td>90.082500</td>
      <td>54.810000</td>
      <td>38.082500</td>
      <td>58.845000</td>
      <td>12.862807</td>
      <td>31.465866</td>
    </tr>
    <tr>
      <th>max</th>
      <td>585.540000</td>
      <td>1.682113e+07</td>
      <td>95.500000</td>
      <td>76.900000</td>
      <td>59.600000</td>
      <td>92.660000</td>
      <td>39.930000</td>
      <td>89.540000</td>
      <td>78.120000</td>
      <td>25.540000</td>
      <td>...</td>
      <td>88.980000</td>
      <td>7.400000</td>
      <td>99.870000</td>
      <td>99.830000</td>
      <td>99.920000</td>
      <td>100.000000</td>
      <td>95.180000</td>
      <td>98.300000</td>
      <td>33.886917</td>
      <td>57.552152</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>




```python
#unique years
Malaria.Year.unique()
```




    array(['2007-01-01T00:00:00.000000000', '2008-01-01T00:00:00.000000000',
           '2009-01-01T00:00:00.000000000', '2010-01-01T00:00:00.000000000',
           '2011-01-01T00:00:00.000000000', '2012-01-01T00:00:00.000000000',
           '2013-01-01T00:00:00.000000000', '2014-01-01T00:00:00.000000000',
           '2015-01-01T00:00:00.000000000', '2016-01-01T00:00:00.000000000',
           '2017-01-01T00:00:00.000000000'], dtype='datetime64[ns]')




```python
#unique countries
Malaria['Country Name'].unique()
```




    array(['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso',
           'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic',
           'Chad', 'Comoros', 'Congo, Dem. Rep.', 'Congo, Rep.',
           "Cote d'Ivoire", 'Djibouti', 'Egypt, Arab Rep.',
           'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon',
           'Gambia, The', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya',
           'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali',
           'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia',
           'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal',
           'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa',
           'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda',
           'Zambia', 'Zimbabwe'], dtype=object)




```python
#the length of the dataset
length_dataset = len(Malaria)
print(length_dataset)
```

    594
    


```python
#count the missing values for each column
missing_values_count = Malaria.isnull().sum()
print(missing_values_count)
```

    Country Name                                                                                 0
    Year                                                                                         0
    Country Code                                                                                 0
    Incidence of malaria (per 1,000 population at risk)                                         44
    Malaria cases reported                                                                      44
    Use of insecticide-treated bed nets (% of under-5 population)                              462
    Children with fever receiving antimalarial drugs (% of children under age 5 with fever)    472
    Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)      488
    People using safely managed drinking water services (% of population)                      495
    People using safely managed drinking water services, rural (% of rural population)         506
    People using safely managed drinking water services, urban (% of urban population)         418
    People using safely managed sanitation services (% of population)                          462
    People using safely managed sanitation services, rural (% of rural population)             484
    People using safely managed sanitation services, urban  (% of urban population)            462
    Rural population (% of total population)                                                     6
    Rural population growth (annual %)                                                           6
    Urban population (% of total population)                                                     6
    Urban population growth (annual %)                                                           6
    People using at least basic drinking water services (% of population)                        6
    People using at least basic drinking water services, rural (% of rural population)          28
    People using at least basic drinking water services, urban (% of urban population)          28
    People using at least basic sanitation services (% of population)                            6
    People using at least basic sanitation services, rural (% of rural population)              28
    People using at least basic sanitation services, urban  (% of urban population)             28
    latitude                                                                                     0
    longitude                                                                                    0
    geometry                                                                                     0
    dtype: int64
    


```python
#count missing values for malaria cases reported per year
Malaria['Malaria cases reported'].isnull().groupby(Malaria['Year']).sum()
```




    Year
    2007-01-01    4
    2008-01-01    4
    2009-01-01    4
    2010-01-01    4
    2011-01-01    4
    2012-01-01    4
    2013-01-01    4
    2014-01-01    4
    2015-01-01    4
    2016-01-01    4
    2017-01-01    4
    Name: Malaria cases reported, dtype: int64




```python
#drop countries with missing values for no incidence of Malaria
new_malaria_data_v1 = Malaria[Malaria['Incidence of malaria (per 1,000 population at risk)'].notna()]
new_malaria_data_v1 = new_malaria_data_v1[new_malaria_data_v1['Incidence of malaria (per 1,000 population at risk)'] != 0]
print(new_malaria_data_v1.head(5))

#also drop countries with missing values or null values for malaria cases reported (after dropping no incidence of malaria)
new_malaria_data_v2 = Malaria[Malaria['Malaria cases reported'].notna()]
new_malaria_data_v2 = new_malaria_data_v2[new_malaria_data_v2['Malaria cases reported'] != 0]
print(new_malaria_data_v2.head(5))
```

       Country Name       Year Country Code  \
    0       Algeria 2007-01-01          DZA   
    1        Angola 2007-01-01          AGO   
    2         Benin 2007-01-01          BEN   
    3      Botswana 2007-01-01          BWA   
    4  Burkina Faso 2007-01-01          BFA   
    
       Incidence of malaria (per 1,000 population at risk)  \
    0                                               0.01     
    1                                             286.72     
    2                                             480.24     
    3                                               1.03     
    4                                             503.80     
    
       Malaria cases reported  \
    0                    26.0   
    1               1533485.0   
    2                     0.0   
    3                   390.0   
    4                 44246.0   
    
       Use of insecticide-treated bed nets (% of under-5 population)  \
    0                                                NaN               
    1                                               18.0               
    2                                                NaN               
    3                                                NaN               
    4                                                NaN               
    
       Children with fever receiving antimalarial drugs (% of children under age 5 with fever)  \
    0                                                NaN                                         
    1                                               29.8                                         
    2                                                NaN                                         
    3                                                NaN                                         
    4                                                NaN                                         
    
       Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)  \
    0                                                NaN                                       
    1                                                1.5                                       
    2                                                NaN                                       
    3                                                NaN                                       
    4                                                NaN                                       
    
       People using safely managed drinking water services (% of population)  \
    0                                                NaN                       
    1                                                NaN                       
    2                                                NaN                       
    3                                                NaN                       
    4                                                NaN                       
    
       People using safely managed drinking water services, rural (% of rural population)  \
    0                                                NaN                                    
    1                                                NaN                                    
    2                                                NaN                                    
    3                                                NaN                                    
    4                                                NaN                                    
    
       ...  Urban population growth (annual %)  \
    0  ...                                2.71   
    1  ...                                5.01   
    2  ...                                4.09   
    3  ...                                4.80   
    4  ...                                5.91   
    
       People using at least basic drinking water services (% of population)  \
    0                                              91.68                       
    1                                              47.96                       
    2                                              63.78                       
    3                                              78.89                       
    4                                              52.27                       
    
       People using at least basic drinking water services, rural (% of rural population)  \
    0                                              85.83                                    
    1                                              23.77                                    
    2                                              54.92                                    
    3                                              57.60                                    
    4                                              45.13                                    
    
       People using at least basic drinking water services, urban (% of urban population)  \
    0                                              94.78                                    
    1                                              65.83                                    
    2                                              76.24                                    
    3                                              94.35                                    
    4                                              76.15                                    
    
       People using at least basic sanitation services (% of population)  \
    0                                              85.85                   
    1                                              37.26                   
    2                                              11.80                   
    3                                              61.60                   
    4                                              15.60                   
    
       People using at least basic sanitation services, rural (% of rural population)  \
    0                                              76.94                                
    1                                              14.00                                
    2                                               4.29                                
    3                                              39.99                                
    4                                               6.38                                
    
       People using at least basic sanitation services, urban  (% of urban population)  \
    0                                              90.57                                 
    1                                              54.44                                 
    2                                              22.36                                 
    3                                              77.30                                 
    4                                              46.49                                 
    
        latitude  longitude                            geometry  
    0  28.033886   1.659626          POINT (28.033886 1.659626)  
    1 -11.202692  17.873887        POINT (-11.202692 17.873887)  
    2   9.307690   2.315834  POINT (9.307689999999999 2.315834)  
    3 -22.328474  24.684866        POINT (-22.328474 24.684866)  
    4  12.238333  -1.561593         POINT (12.238333 -1.561593)  
    
    [5 rows x 27 columns]
       Country Name       Year Country Code  \
    0       Algeria 2007-01-01          DZA   
    1        Angola 2007-01-01          AGO   
    3      Botswana 2007-01-01          BWA   
    4  Burkina Faso 2007-01-01          BFA   
    5       Burundi 2007-01-01          BDI   
    
       Incidence of malaria (per 1,000 population at risk)  \
    0                                               0.01     
    1                                             286.72     
    3                                               1.03     
    4                                             503.80     
    5                                             222.92     
    
       Malaria cases reported  \
    0                    26.0   
    1               1533485.0   
    3                   390.0   
    4                 44246.0   
    5               1101644.0   
    
       Use of insecticide-treated bed nets (% of under-5 population)  \
    0                                                NaN               
    1                                               18.0               
    3                                                NaN               
    4                                                NaN               
    5                                                NaN               
    
       Children with fever receiving antimalarial drugs (% of children under age 5 with fever)  \
    0                                                NaN                                         
    1                                               29.8                                         
    3                                                NaN                                         
    4                                                NaN                                         
    5                                                NaN                                         
    
       Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)  \
    0                                                NaN                                       
    1                                                1.5                                       
    3                                                NaN                                       
    4                                                NaN                                       
    5                                                NaN                                       
    
       People using safely managed drinking water services (% of population)  \
    0                                                NaN                       
    1                                                NaN                       
    3                                                NaN                       
    4                                                NaN                       
    5                                                NaN                       
    
       People using safely managed drinking water services, rural (% of rural population)  \
    0                                                NaN                                    
    1                                                NaN                                    
    3                                                NaN                                    
    4                                                NaN                                    
    5                                                NaN                                    
    
       ...  Urban population growth (annual %)  \
    0  ...                                2.71   
    1  ...                                5.01   
    3  ...                                4.80   
    4  ...                                5.91   
    5  ...                                5.82   
    
       People using at least basic drinking water services (% of population)  \
    0                                              91.68                       
    1                                              47.96                       
    3                                              78.89                       
    4                                              52.27                       
    5                                              54.73                       
    
       People using at least basic drinking water services, rural (% of rural population)  \
    0                                              85.83                                    
    1                                              23.77                                    
    3                                              57.60                                    
    4                                              45.13                                    
    5                                              51.39                                    
    
       People using at least basic drinking water services, urban (% of urban population)  \
    0                                              94.78                                    
    1                                              65.83                                    
    3                                              94.35                                    
    4                                              76.15                                    
    5                                              85.24                                    
    
       People using at least basic sanitation services (% of population)  \
    0                                              85.85                   
    1                                              37.26                   
    3                                              61.60                   
    4                                              15.60                   
    5                                              45.91                   
    
       People using at least basic sanitation services, rural (% of rural population)  \
    0                                              76.94                                
    1                                              14.00                                
    3                                              39.99                                
    4                                               6.38                                
    5                                              46.26                                
    
       People using at least basic sanitation services, urban  (% of urban population)  \
    0                                              90.57                                 
    1                                              54.44                                 
    3                                              77.30                                 
    4                                              46.49                                 
    5                                              42.78                                 
    
        latitude  longitude                      geometry  
    0  28.033886   1.659626    POINT (28.033886 1.659626)  
    1 -11.202692  17.873887  POINT (-11.202692 17.873887)  
    3 -22.328474  24.684866  POINT (-22.328474 24.684866)  
    4  12.238333  -1.561593   POINT (12.238333 -1.561593)  
    5  -3.373056  29.918886   POINT (-3.373056 29.918886)  
    
    [5 rows x 27 columns]
    


```python
#Scatterplot Incidence of Malaria and Malaria Cases Reported
plt.scatter(x=Malaria['Incidence of malaria (per 1,000 population at risk)'],y=Malaria['Malaria cases reported'],color='red')

plt.title("Correlation Incidence of Malaria and Malaria Cases Reported")
plt.xlabel("Malaria Cases Reported")
plt.ylabel("Incidence of Malaria")
plt.show()
```


    
![output_24_0](https://user-images.githubusercontent.com/75635908/166151004-5b7f2b3d-360f-487b-a6e6-247eb7d5a2bb.png)

    



```python
#Trend in cases of Malaria in Africa each year
Malaria1 = Malaria[['Year','Malaria cases reported']]
Malaria1 = Malaria.rename(columns={'Malaria cases reported':'Malaria_cases_reported'})
Malaria_cases_yearly = Malaria1.groupby(Malaria1.Year).Malaria_cases_reported.sum()
Malaria_cases_yearly.reset_index()
Malaria_cases_yearly = Malaria_cases_yearly.to_frame()

#plot over the years
sns.lineplot(data=Malaria_cases_yearly.Malaria_cases_reported)

plt.title("Malaria cases in Africa per Year")
plt.xlabel("Year")
plt.ylabel("Malaria cases")
plt.show()
```


    
![output_25_0](https://user-images.githubusercontent.com/75635908/166150978-8f71aeac-bbbb-4f62-ad65-8c1d4f8195f4.png)

    



```python
#count the least missing values per country for the use of insecticide-treated bed net
Malaria['Use of insecticide-treated bed nets (% of under-5 population)'].isnull().groupby(Malaria['Country Name']).sum().sort_values()
```




    Country Name
    Senegal                      4
    Nigeria                      4
    Rwanda                       5
    Tanzania                     6
    Malawi                       6
    Zambia                       6
    Sierra Leone                 6
    Zimbabwe                     7
    Niger                        7
    Mozambique                   7
    Madagascar                   7
    Liberia                      7
    Kenya                        7
    Guinea                       7
    Ghana                        7
    Uganda                       7
    Togo                         8
    Ethiopia                     8
    Namibia                      8
    Gambia, The                  8
    Burundi                      8
    Congo, Dem. Rep.             8
    Mali                         8
    Angola                       8
    Sao Tome and Principe        9
    Benin                        9
    Burkina Faso                 9
    Mauritania                   9
    Cameroon                     9
    South Sudan                  9
    Chad                         9
    Guinea-Bissau                9
    Congo, Rep.                  9
    Cote d'Ivoire                9
    Gabon                        9
    Eswatini                     9
    Eritrea                      9
    Comoros                     10
    Sudan                       10
    Botswana                    10
    Equatorial Guinea           10
    Djibouti                    10
    Central African Republic    10
    Tunisia                     11
    South Africa                11
    Algeria                     11
    Seychelles                  11
    Morocco                     11
    Mauritius                   11
    Libya                       11
    Egypt, Arab Rep.            11
    Cabo Verde                  11
    Somalia                     11
    Lesotho                     11
    Name: Use of insecticide-treated bed nets (% of under-5 population), dtype: int64




```python
#regplot
#drop missing values first
Malaria_treated_nets = Malaria[Malaria['Use of insecticide-treated bed nets (% of under-5 population)'].notna()]
Malaria_antimalarial_medication = Malaria[Malaria['Children with fever receiving antimalarial drugs (% of children under age 5 with fever)'].notna()]
Malaria_IPT = Malaria[Malaria['Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)'].notna()]

#plotting for malaria cases reported
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,20))

sns.regplot(x=Malaria_treated_nets['Use of insecticide-treated bed nets (% of under-5 population)'],y=Malaria_treated_nets['Malaria cases reported'],data=Malaria_treated_nets,ax=ax1)
sns.regplot(x=Malaria_antimalarial_medication['Children with fever receiving antimalarial drugs (% of children under age 5 with fever)'],y=Malaria_antimalarial_medication['Malaria cases reported'],data=Malaria_antimalarial_medication,ax=ax2)
sns.regplot(x=Malaria_IPT['Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)'],y=Malaria_IPT['Malaria cases reported'],data=Malaria_IPT,ax=ax3)

plt.show()
```


    
![output_27_0](https://user-images.githubusercontent.com/75635908/166150952-aa73702a-f24e-48cc-9fbf-f3b399a26cb0.png)




```python
#plotting for malaria incidence risk
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,20))

sns.regplot(x=Malaria_treated_nets['Use of insecticide-treated bed nets (% of under-5 population)'],y=Malaria_treated_nets['Incidence of malaria (per 1,000 population at risk)'],data=Malaria_treated_nets,ax=ax1)
sns.regplot(x=Malaria_antimalarial_medication['Children with fever receiving antimalarial drugs (% of children under age 5 with fever)'],y=Malaria_antimalarial_medication['Incidence of malaria (per 1,000 population at risk)'],data=Malaria_antimalarial_medication,ax=ax2)
sns.regplot(x=Malaria_IPT['Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)'],y=Malaria_IPT['Incidence of malaria (per 1,000 population at risk)'],data=Malaria_IPT,ax=ax3)

plt.show()
```


    
![output_28_0](https://user-images.githubusercontent.com/75635908/166150934-d075b63b-f64e-4e0f-b242-6f12925c3410.png)



```python
#distribution plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,20))

sns.distplot(Malaria_treated_nets['Use of insecticide-treated bed nets (% of under-5 population)'],kde=False, ax=ax1)
sns.distplot(Malaria_antimalarial_medication['Children with fever receiving antimalarial drugs (% of children under age 5 with fever)'],kde=False,ax=ax2)
sns.distplot(Malaria_IPT['Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)'],kde=False,ax=ax3)

plt.show()
```

    D:\DataScienceProjects\Malaria_In_Africa\Malaria_In_Africa\lib\site-packages\seaborn\distributions.py:2619: FutureWarning:
    
    `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
    
    


    

![output_29_1](https://user-images.githubusercontent.com/75635908/166150902-fddc1117-8e83-4a6f-b39e-c5175153b766.png)
    

