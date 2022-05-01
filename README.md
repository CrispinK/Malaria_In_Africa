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
#Incidence of Malaria at risk
fig1 = px.choropleth(Malaria_data,locations=Malaria_data['Country Code'],color=Malaria_data['Incidence of malaria (per 1,000 population at risk)'],color_continuous_scale='Blues',locationmode='ISO-3',scope='africa',animation_frame=Malaria_data['Year'],title="Incidence of Malaria at risk in Africa",labels={'color':'Incidence of Malaria'})

fig1.show()
```


<div>                            <div id="74b10e59-82d4-4587-8f79-3222ddbd15b6" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("74b10e59-82d4-4587-8f79-3222ddbd15b6")) {                    Plotly.newPlot(                        "74b10e59-82d4-4587-8f79-3222ddbd15b6",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.01,286.72,480.24,1.03,503.8,222.92,0.15,386.68,471.11,210.71,60.65,481.93,281.83,509.98,3.84,0.0,398.15,12.54,0.29,121.69,110.66,316.01,322.33,343.69,72.81,78.02,null,383.18,0.0,22.78,370.08,388.81,70.77,null,0.0,399.45,12.97,386.68,421.33,90.64,14.56,101.91,null,379.94,100.56,1.29,276.75,41.08,184.53,434.49,null,377.94,195.74,175.12],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"africa"},"coloraxis":{"colorbar":{"title":{"text":"Incidence of malaria (per 1,000 population at risk)"}},"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]]},"legend":{"tracegroupgap":0},"title":{"text":"Incidence of Malaria at risk in Africa"},"updatemenus":[{"buttons":[{"args":[null,{"frame":{"duration":500,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":500,"easing":"linear"}}],"label":"&#9654;","method":"animate"},{"args":[[null],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"&#9724;","method":"animate"}],"direction":"left","pad":{"r":10,"t":70},"showactive":false,"type":"buttons","x":0.1,"xanchor":"right","y":0,"yanchor":"top"}],"sliders":[{"active":0,"currentvalue":{"prefix":"Year="},"len":0.9,"pad":{"b":10,"t":60},"steps":[{"args":[["2007-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2007-01-01","method":"animate"},{"args":[["2008-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2008-01-01","method":"animate"},{"args":[["2009-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2009-01-01","method":"animate"},{"args":[["2010-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2010-01-01","method":"animate"},{"args":[["2011-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2011-01-01","method":"animate"},{"args":[["2012-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2012-01-01","method":"animate"},{"args":[["2013-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2013-01-01","method":"animate"},{"args":[["2014-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2014-01-01","method":"animate"},{"args":[["2015-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2015-01-01","method":"animate"},{"args":[["2016-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2016-01-01","method":"animate"},{"args":[["2017-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2017-01-01","method":"animate"}],"x":0.1,"xanchor":"left","y":0,"yanchor":"top"}]},                        {"responsive": true}                    ).then(function(){
                            Plotly.addFrames('74b10e59-82d4-4587-8f79-3222ddbd15b6', [{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.01,286.72,480.24,1.03,503.8,222.92,0.15,386.68,471.11,210.71,60.65,481.93,281.83,509.98,3.84,0.0,398.15,12.54,0.29,121.69,110.66,316.01,322.33,343.69,72.81,78.02,null,383.18,0.0,22.78,370.08,388.81,70.77,null,0.0,399.45,12.97,386.68,421.33,90.64,14.56,101.91,null,379.94,100.56,1.29,276.75,41.08,184.53,434.49,null,377.94,195.74,175.12],"type":"choropleth"}],"name":"2007-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2008-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","COM","TCD","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,241.19,453.17,1.93,533.39,222.77,0.28,358.72,461.79,59.21,215.1,471.43,249.16,510.08,3.74,0.0,378.24,6.78,0.2,90.66,118.87,306.84,333.72,370.17,89.02,67.68,null,383.33,0.0,23.47,392.0,379.06,61.14,null,0.0,399.79,7.46,409.1,424.66,72.03,36.57,86.42,null,413.85,63.1,1.57,267.14,35.05,171.0,379.9,null,410.46,176.72,74.73],"type":"choropleth"}],"name":"2008-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2009-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,204.35,417.96,2.1,552.03,220.65,0.51,322.98,448.45,222.91,57.81,452.18,226.32,494.48,4.33,0.0,351.29,8.86,0.36,123.35,145.95,297.98,351.39,398.14,113.8,67.94,null,368.81,0.0,42.92,394.29,374.56,31.64,null,0.0,402.3,3.36,419.1,416.59,158.16,35.15,57.12,null,442.15,38.44,1.2,255.07,32.29,157.11,326.88,null,433.79,172.0,90.5],"type":"choropleth"}],"name":"2009-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2010-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,185.52,387.76,1.69,551.24,210.2,0.37,295.53,434.51,225.94,52.98,428.31,220.92,469.27,1.6,0.0,339.99,26.33,0.9,128.4,177.82,289.27,364.15,414.66,134.37,67.71,null,345.77,0.0,42.24,386.02,383.6,38.83,null,0.0,398.41,1.54,425.64,398.9,126.31,15.19,59.28,null,458.74,29.59,1.57,243.97,30.66,145.46,308.88,null,417.35,177.02,109.43],"type":"choropleth"}],"name":"2010-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2011-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,175.99,370.11,0.51,539.56,184.15,0.05,265.1,419.33,210.48,35.18,399.59,224.38,442.11,3.42,0.0,342.4,23.86,1.83,116.13,212.72,257.26,379.52,426.9,140.55,67.86,null,330.41,0.0,36.55,362.78,404.73,47.58,null,0.0,389.92,2.13,427.88,372.56,39.29,45.75,49.91,null,453.65,24.35,1.9,235.87,29.97,132.48,313.4,null,385.7,186.7,70.68],"type":"choropleth"}],"name":"2011-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2012-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.03,174.43,378.03,0.22,527.54,153.93,0.01,245.13,413.09,195.34,68.85,363.0,224.63,396.28,3.31,0.0,357.75,16.15,1.86,116.19,245.53,274.82,374.29,427.8,128.75,73.36,null,307.9,0.0,71.36,314.02,435.65,28.42,null,0.0,381.78,3.36,430.51,347.74,71.46,56.8,56.92,null,447.47,24.45,1.25,232.68,30.16,116.25,349.7,null,331.77,203.08,57.22],"type":"choropleth"}],"name":"2012-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2013-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,180.9,395.0,0.53,487.52,140.59,0.17,243.04,406.82,179.35,71.69,327.62,231.03,338.87,2.54,0.0,365.75,15.03,3.16,111.6,272.83,236.99,349.28,408.08,113.39,82.48,null,317.28,0.0,65.21,267.86,452.82,33.22,null,0.0,376.98,4.55,420.49,328.65,121.45,48.12,67.9,null,432.7,28.05,1.61,234.38,31.45,111.78,385.39,null,254.24,225.77,81.95],"type":"choropleth"}],"name":"2013-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2014-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,187.94,399.24,1.5,436.06,141.51,0.19,244.08,393.04,170.27,2.9,308.39,231.85,315.07,14.01,0.0,360.95,33.12,2.32,57.1,285.74,142.02,315.76,381.09,93.9,83.87,null,337.57,0.0,45.78,237.0,441.01,49.2,null,0.0,364.83,14.49,400.25,314.4,219.81,8.96,39.51,null,409.31,32.1,2.15,236.14,33.38,118.94,384.68,null,220.62,222.98,101.89],"type":"choropleth"}],"name":"2014-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2015-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,199.99,411.82,0.37,400.09,165.5,0.05,254.5,379.91,167.28,1.67,308.83,226.71,320.04,13.82,0.0,339.48,19.2,0.51,52.77,284.44,195.04,277.19,356.64,79.77,72.17,null,346.97,0.0,97.32,217.04,391.83,61.61,null,0.0,355.88,10.88,369.83,296.08,341.96,10.32,69.8,null,403.72,37.27,0.21,240.36,35.88,121.74,364.31,null,236.11,202.55,97.65],"type":"choropleth"}],"name":"2015-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2016-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,219.99,421.58,0.81,401.73,225.74,0.35,264.31,362.02,169.93,1.44,322.77,233.38,354.66,19.81,0.0,306.97,25.64,1.12,41.41,270.67,116.53,236.0,331.47,71.35,70.38,null,386.3,0.0,56.58,210.66,384.22,71.5,null,0.0,344.82,22.12,358.74,281.38,585.54,11.01,45.66,null,391.33,37.26,0.77,244.55,41.73,123.57,324.86,null,304.41,181.39,65.78],"type":"choropleth"}],"name":"2016-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2017-01-01<br>Country Code=%{location}<br>Incidence of malaria (per 1,000 population at risk)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,228.91,399.56,2.05,399.94,250.27,3.03,262.23,347.33,172.3,3.97,329.13,240.64,362.36,20.72,0.0,285.72,33.97,2.3,36.74,254.24,53.02,212.56,311.53,78.33,70.1,null,401.11,0.0,75.67,216.26,386.78,55.49,null,0.0,326.4,46.75,356.57,283.06,538.34,10.81,52.35,null,364.13,37.13,3.95,245.8,46.75,123.96,278.2,null,336.76,160.05,108.55],"type":"choropleth"}],"name":"2017-01-01"}]);
                        }).then(function(){

var gd = document.getElementById('74b10e59-82d4-4587-8f79-3222ddbd15b6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#Malaria Cases Reported by country
fig2 = px.choropleth(Malaria_data,locations=Malaria_data['Country Code'],color=Malaria_data['Malaria cases reported'],color_continuous_scale='Blues',locationmode='ISO-3',scope='africa',animation_frame=Malaria_data['Year'],title="Malaria Cases in Africa",labels={'color':'number of Malaria cases'})

fig2.show()
```


<div>                            <div id="562c8536-58a8-4c3e-9fb4-475aa03a0474" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("562c8536-58a8-4c3e-9fb4-475aa03a0474")) {                    Plotly.newPlot(                        "562c8536-58a8-4c3e-9fb4-475aa03a0474",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[26.0,1533485.0,0.0,390.0,44246.0,1101644.0,18.0,0.0,0.0,48288.0,38913.0,740858.0,103213.0,0.0,2320.0,0.0,6287.0,15565.0,84.0,451816.0,45186.0,0.0,476484.0,44518.0,14284.0,0.0,null,492272.0,0.0,48497.0,0.0,0.0,0.0,null,0.0,141663.0,4242.0,268164.0,0.0,382686.0,2421.0,118332.0,null,0.0,16675.0,6327.0,0.0,686908.0,1845917.0,258716.0,null,1045378.0,0.0,116518.0],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"africa"},"coloraxis":{"colorbar":{"title":{"text":"Malaria cases reported"}},"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]]},"legend":{"tracegroupgap":0},"title":{"text":"Malaria Cases in Africa"},"updatemenus":[{"buttons":[{"args":[null,{"frame":{"duration":500,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":500,"easing":"linear"}}],"label":"&#9654;","method":"animate"},{"args":[[null],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"&#9724;","method":"animate"}],"direction":"left","pad":{"r":10,"t":70},"showactive":false,"type":"buttons","x":0.1,"xanchor":"right","y":0,"yanchor":"top"}],"sliders":[{"active":0,"currentvalue":{"prefix":"Year="},"len":0.9,"pad":{"b":10,"t":60},"steps":[{"args":[["2007-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2007-01-01","method":"animate"},{"args":[["2008-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2008-01-01","method":"animate"},{"args":[["2009-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2009-01-01","method":"animate"},{"args":[["2010-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2010-01-01","method":"animate"},{"args":[["2011-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2011-01-01","method":"animate"},{"args":[["2012-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2012-01-01","method":"animate"},{"args":[["2013-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2013-01-01","method":"animate"},{"args":[["2014-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2014-01-01","method":"animate"},{"args":[["2015-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2015-01-01","method":"animate"},{"args":[["2016-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2016-01-01","method":"animate"},{"args":[["2017-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2017-01-01","method":"animate"}],"x":0.1,"xanchor":"left","y":0,"yanchor":"top"}]},                        {"responsive": true}                    ).then(function(){
                            Plotly.addFrames('562c8536-58a8-4c3e-9fb4-475aa03a0474', [{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[26.0,1533485.0,0.0,390.0,44246.0,1101644.0,18.0,0.0,0.0,48288.0,38913.0,740858.0,103213.0,0.0,2320.0,0.0,6287.0,15565.0,84.0,451816.0,45186.0,0.0,476484.0,44518.0,14284.0,0.0,null,492272.0,0.0,48497.0,0.0,0.0,0.0,null,0.0,141663.0,4242.0,268164.0,0.0,382686.0,2421.0,118332.0,null,0.0,16675.0,6327.0,0.0,686908.0,1845917.0,258716.0,null,1045378.0,0.0,116518.0],"type":"choropleth"}],"name":"2007-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2008-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","COM","TCD","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[3.0,1377992.0,0.0,927.0,36514.0,876741.0,35.0,0.0,0.0,38917.0,47757.0,2270.0,117291.0,3527.0,2289.0,0.0,9503.0,8764.0,58.0,458561.0,40701.0,39164.0,1094483.0,33405.0,11299.0,839903.0,null,606952.0,0.0,93234.0,0.0,0.0,302.0,null,0.0,120259.0,1092.0,682685.0,143079.0,316242.0,6258.0,241926.0,null,176356.0,36905.0,7796.0,52011.0,569296.0,4508.0,397283.0,null,979298.0,0.0,32788.0],"type":"choropleth"}],"name":"2008-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2009-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,1573422.0,889597.0,1024.0,182527.0,1185622.0,65.0,0.0,0.0,0.0,38920.0,1879694.0,92855.0,7388.0,2686.0,0.0,14184.0,11759.0,106.0,1144640.0,660.0,50378.0,1104370.0,35841.0,11757.0,0.0,null,839581.0,0.0,215110.0,0.0,0.0,940.0,null,0.0,93874.0,505.0,309675.0,479845.0,698745.0,6182.0,165933.0,null,646808.0,25202.0,6072.0,0.0,711462.0,211.0,571611.0,null,1301337.0,0.0,114028.0],"type":"choropleth"}],"name":"2009-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2010-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[1.0,1682870.0,0.0,1046.0,804539.0,1763447.0,47.0,0.0,0.0,200448.0,36538.0,2417780.0,0.0,62726.0,1010.0,0.0,53813.0,35982.0,268.0,1158197.0,13936.0,116353.0,1071637.0,20936.0,50391.0,898531.0,null,922173.0,0.0,202450.0,0.0,239787.0,6367.0,null,0.0,1522577.0,556.0,642774.0,551187.0,669322.0,2740.0,330331.0,null,934028.0,24833.0,8060.0,900283.0,720557.0,1278998.0,1006702.0,null,1581160.0,0.0,249379.0],"type":"choropleth"}],"name":"2010-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2011-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[1.0,1632282.0,68745.0,432.0,428113.0,1575237.0,7.0,33086.0,0.0,181126.0,24856.0,4561981.0,37744.0,29976.0,2189.0,0.0,22466.0,34848.0,549.0,1480306.0,0.0,268020.0,1041260.0,95574.0,71982.0,1002805.0,null,1921159.0,0.0,224498.0,304499.0,307035.0,5991.0,null,0.0,1756874.0,1860.0,838585.0,0.0,273293.0,8442.0,274119.0,null,638859.0,3351.0,9866.0,112024.0,506806.0,2150761.0,519450.0,null,231873.0,0.0,319935.0],"type":"choropleth"}],"name":"2011-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2012-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[55.0,1496834.0,705839.0,193.0,3858046.0,2166690.0,1.0,66656.0,87566.0,7710.0,49840.0,4791598.0,120319.0,1140627.0,2153.0,0.0,15169.0,21815.0,562.0,1692578.0,19753.0,313469.0,3755166.0,340258.0,50381.0,1453471.0,null,1412629.0,0.0,402900.0,1564984.0,968136.0,9037.0,null,0.0,1853276.0,194.0,2329260.0,0.0,563852.0,10701.0,280241.0,null,1537322.0,35712.0,6621.0,225371.0,526931.0,1986955.0,909129.0,null,2662258.0,0.0,276963.0],"type":"choropleth"}],"name":"2012-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2013-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[8.0,1999868.0,1090602.0,456.0,3769051.0,4178338.0,22.0,69232.0,163701.0,754565.0,53156.0,6719887.0,43232.0,2524326.0,1684.0,0.0,16405.0,21317.0,962.0,2645454.0,28982.0,242513.0,1639451.0,211257.0,54584.0,2375129.0,null,1244220.0,0.0,433450.0,1280892.0,1506940.0,13085.0,null,0.0,3282172.0,4911.0,2373591.0,0.0,1040557.0,9243.0,366687.0,null,1701958.0,8944.0,8645.0,262520.0,592383.0,1550250.0,965334.0,null,1502362.0,0.0,422633.0],"type":"choropleth"}],"name":"2013-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2014-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,2298979.0,1130558.0,1346.0,5428655.0,4726299.0,26.0,0.0,295088.0,914032.0,2203.0,10288519.0,66323.0,3712831.0,9439.0,0.0,20417.0,50534.0,711.0,2118815.0,31900.0,168256.0,3415912.0,660207.0,93431.0,2851555.0,null,881224.0,0.0,468743.0,2905310.0,2220956.0,15835.0,null,0.0,7407175.0,15914.0,2010489.0,7826954.0,1719904.0,1754.0,268912.0,null,1374476.0,11001.0,11705.0,71377.0,1068506.0,680442.0,1524339.0,null,3631939.0,4077547.0,548276.0],"type":"choropleth"}],"name":"2014-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2015-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,2769305.0,1524739.0,326.0,7015446.0,5428710.0,7.0,1193281.0,598833.0,787046.0,1300.0,12538805.0,51529.0,3375904.0,9473.0,0.0,15142.0,28036.0,157.0,1867059.0,23867.0,245435.0,4319919.0,810979.0,146027.0,1581168.0,null,941711.0,0.0,937241.0,3661238.0,2454508.0,22631.0,null,0.0,8222814.0,12168.0,2392108.0,7131972.0,2694566.0,2058.0,492253.0,null,1483376.0,20953.0,1157.0,24371.0,586827.0,4241364.0,1508015.0,null,7137662.0,4184661.0,482379.0],"type":"choropleth"}],"name":"2015-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2016-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,3794253.0,1395618.0,716.0,9779411.0,8793176.0,48.0,1694002.0,1032764.0,1294768.0,1143.0,16821130.0,171847.0,3645081.0,13804.0,0.0,147714.0,24251.0,350.0,1718504.0,23915.0,159997.0,4505442.0,992146.0,152404.0,2931406.0,null,1191137.0,0.0,655480.0,4827373.0,2311098.0,23042.0,null,0.0,9690873.0,25198.0,4258110.0,12293820.0,4725577.0,2238.0,349540.0,null,1775306.0,35628.0,4323.0,7619.0,575015.0,5188863.0,1746234.0,null,9385132.0,4851319.0,314003.0],"type":"choropleth"}],"name":"2016-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2017-01-01<br>Country Code=%{location}<br>Malaria cases reported=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[0.0,3874892.0,1774022.0,1900.0,10255415.0,8453810.0,423.0,1317371.0,383309.0,1962372.0,3230.0,16793002.0,127939.0,3475953.0,14671.0,0.0,15725.0,54005.0,724.0,1530739.0,35244.0,72412.0,5584185.0,1335323.0,92846.0,3419883.0,null,1783968.0,0.0,935229.0,4901344.0,2097797.0,20105.0,null,0.0,9892601.0,54268.0,2761268.0,11639713.0,4413473.0,2239.0,395706.0,null,1651236.0,35138.0,22517.0,1488005.0,720879.0,5354819.0,1755577.0,null,11667831.0,5505639.0,467508.0],"type":"choropleth"}],"name":"2017-01-01"}]);
                        }).then(function(){

var gd = document.getElementById('562c8536-58a8-4c3e-9fb4-475aa03a0474');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#Use of insecticide-treated bed nets
fig3 = px.choropleth(Malaria_data,locations=Malaria_data['Country Code'],color=Malaria_data['Use of insecticide-treated bed nets (% of under-5 population)'],color_continuous_scale='Blues',locationmode='ISO-3',scope='africa',animation_frame=Malaria_data['Year'],title="Malaria in Africa: Use of Insecticide-treated Bed Nets",labels={'color':'Use of insecticide-treated bed nets'})

fig3.show()
```


<div>                            <div id="fc81547c-48d3-41bb-bbdb-4f3a568687dc" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("fc81547c-48d3-41bb-bbdb-4f3a568687dc")) {                    Plotly.newPlot(                        "fc81547c-48d3-41bb-bbdb-4f3a568687dc",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,18.0,null,null,null,null,null,null,null,null,null,6.0,null,null,null,null,null,null,1.0,33.0,null,null,null,5.0,null,null,null,null,null,null,null,null,null,null,null,7.0,11.0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,29.0,null],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"africa"},"coloraxis":{"colorbar":{"title":{"text":"Use of insecticide-treated bed nets (% of under-5 population)"}},"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]]},"legend":{"tracegroupgap":0},"title":{"text":"Malaria in Africa: Use of Insecticide-treated Bed Nets"},"updatemenus":[{"buttons":[{"args":[null,{"frame":{"duration":500,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":500,"easing":"linear"}}],"label":"&#9654;","method":"animate"},{"args":[[null],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"&#9724;","method":"animate"}],"direction":"left","pad":{"r":10,"t":70},"showactive":false,"type":"buttons","x":0.1,"xanchor":"right","y":0,"yanchor":"top"}],"sliders":[{"active":0,"currentvalue":{"prefix":"Year="},"len":0.9,"pad":{"b":10,"t":60},"steps":[{"args":[["2007-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2007-01-01","method":"animate"},{"args":[["2008-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2008-01-01","method":"animate"},{"args":[["2009-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2009-01-01","method":"animate"},{"args":[["2010-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2010-01-01","method":"animate"},{"args":[["2011-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2011-01-01","method":"animate"},{"args":[["2012-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2012-01-01","method":"animate"},{"args":[["2013-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2013-01-01","method":"animate"},{"args":[["2014-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2014-01-01","method":"animate"},{"args":[["2015-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2015-01-01","method":"animate"},{"args":[["2016-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2016-01-01","method":"animate"},{"args":[["2017-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2017-01-01","method":"animate"}],"x":0.1,"xanchor":"left","y":0,"yanchor":"top"}]},                        {"responsive": true}                    ).then(function(){
                            Plotly.addFrames('fc81547c-48d3-41bb-bbdb-4f3a568687dc', [{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,18.0,null,null,null,null,null,null,null,null,null,6.0,null,null,null,null,null,null,1.0,33.0,null,null,null,5.0,null,null,null,null,null,null,null,null,null,null,null,7.0,11.0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,29.0,null],"type":"choropleth"}],"name":"2007-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2008-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","COM","TCD","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,48.9,null,null,55.1,null,28.0,4.5,null,null,null,null,null,null,null,null,null,null,null,22.8,null,null,6.0,56.0,null,null,null,26.0,null,null,null,null,26.0,null,null,null,41.0,null],"type":"choropleth"}],"name":"2008-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2009-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,19.9,null,null,null,null,null,null,null,null,null,null,46.7,null,26.0,null,46.0,null,null,null,null,null,null,34.0,43.0,null,null,56.0,29.0,null,null,null,null,25.3,25.3,null,null,null,32.8,null,17.3],"type":"choropleth"}],"name":"2009-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2010-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,47.4,45.3,null,null,36.4,9.8,null,38.1,null,null,null,null,null,20.4,1.5,null,null,33.3,null,null,35.5,42.2,null,null,null,null,39.4,45.6,null,null,null,null,null,63.7,29.4,69.6,null,null,null,30.3,null,null,null,null,63.6,57.1,null,null,50.0,null],"type":"choropleth"}],"name":"2010-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2011-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,25.9,null,null,null,null,null,21.0,null,null,null,null,null,null,null,null,23.0,null,null,30.1,null,null,39.0,null,null,null,null,37.1,null,76.5,null,null,18.7,null,null,35.7,null,null,16.4,69.6,null,34.5,null,null,null,null,null,null,null,null,null,42.8,null,9.7],"type":"choropleth"}],"name":"2011-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2012-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,69.7,30.9,null,53.8,null,null,null,null,41.1,null,26.3,37.2,null,null,null,null,null,null,38.8,null,null,26.0,null,null,null,null,null,null,56.0,null,null,null,null,null,null,20.1,null,null,null,null,null,null,null,null,null,null,72.0,null,null,null,57.0,null],"type":"choropleth"}],"name":"2012-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2013-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,47.2,null,null,null,null,null,38.1,null,61.5,null,69.0,null,null,null,null,5.6,null,16.6,74.1,null,45.8,null,49.0,null,null,45.8,null,null,null,null,null,null,null],"type":"choropleth"}],"name":"2013-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2014-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,72.7,null,75.3,null,null,54.8,null,null,null,55.8,null,null,null,null,null,null,null,null,null,null,46.6,null,80.6,54.1,null,null,null,null,65.5,null,null,null,null,null,null,null,25.4,null,61.1,43.2,null,null,null,null,null,null,null,42.8,null,null,40.6,26.8],"type":"choropleth"}],"name":"2014-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2015-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,36.4,null,null,60.5,null,null,null,null,null,null,45.3,null,null,null,null,null,56.1,null,null,null,null,null,79.3,32.1,null,null,47.9,null,95.5,43.6,67.7,null,55.4,null,null,null,null,null,null,null,null,null,74.3,null,9.0],"type":"choropleth"}],"name":"2015-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2016-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,21.7,null,null,null,null,null,null,null,null,null,null,null,59.7,null,null,null,null,null,null,null,null,52.2,67.9,null,null,null,43.7,null,73.4,42.7,null,null,null,null,null,null,null,null,null,null,66.6,null,44.1,null,null,null,null,54.4,null,null,62.0,null,null],"type":"choropleth"}],"name":"2016-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2017-01-01<br>Country Code=%{location}<br>Use of insecticide-treated bed nets (% of under-5 population)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,39.9,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,62.4,null,null,null,null,null,null,null,null,67.5,null,null,null,null,null,null,null,49.1,68.0,null,60.7,null,59.5,null,null,null,null,54.6,69.7,null,null,null,null],"type":"choropleth"}],"name":"2017-01-01"}]);
                        }).then(function(){

var gd = document.getElementById('fc81547c-48d3-41bb-bbdb-4f3a568687dc');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#Children with fever receiving antimalarial drugs (% of children under age 5 with fever)
fig4 = px.choropleth(Malaria_data,locations=Malaria_data['Country Code'],color=Malaria_data['Children with fever receiving antimalarial drugs (% of children under age 5 with fever)'],color_continuous_scale='Blues',locationmode='ISO-3',scope='africa',animation_frame=Malaria_data['Year'],title="Malaria in Africa: Children with Fever receiving Antimalarial Drugs",labels={'color':'Children with fever receiving antimalarial drugs'})

fig4.show()
```


<div>                            <div id="9f8fc860-2069-4249-ba26-2b93fb5e7011" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("9f8fc860-2069-4249-ba26-2b93fb5e7011")) {                    Plotly.newPlot(                        "9f8fc860-2069-4249-ba26-2b93fb5e7011",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,29.8,null,null,null,null,null,null,null,null,null,29.8,null,null,null,null,null,null,0.6,10.0,null,null,null,74.0,null,null,null,58.8,null,null,null,null,21.0,null,null,23.0,9.8,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,38.4,null],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"africa"},"coloraxis":{"colorbar":{"title":{"text":"Children with fever receiving antimalarial drugs (% of children under age 5 with fever)"}},"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]]},"legend":{"tracegroupgap":0},"title":{"text":"Malaria in Africa: Children with Fever receiving Antimalarial Drugs"},"updatemenus":[{"buttons":[{"args":[null,{"frame":{"duration":500,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":500,"easing":"linear"}}],"label":"&#9654;","method":"animate"},{"args":[[null],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"&#9724;","method":"animate"}],"direction":"left","pad":{"r":10,"t":70},"showactive":false,"type":"buttons","x":0.1,"xanchor":"right","y":0,"yanchor":"top"}],"sliders":[{"active":0,"currentvalue":{"prefix":"Year="},"len":0.9,"pad":{"b":10,"t":60},"steps":[{"args":[["2007-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2007-01-01","method":"animate"},{"args":[["2008-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2008-01-01","method":"animate"},{"args":[["2009-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2009-01-01","method":"animate"},{"args":[["2010-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2010-01-01","method":"animate"},{"args":[["2011-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2011-01-01","method":"animate"},{"args":[["2012-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2012-01-01","method":"animate"},{"args":[["2013-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2013-01-01","method":"animate"},{"args":[["2014-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2014-01-01","method":"animate"},{"args":[["2015-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2015-01-01","method":"animate"},{"args":[["2016-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2016-01-01","method":"animate"},{"args":[["2017-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2017-01-01","method":"animate"}],"x":0.1,"xanchor":"left","y":0,"yanchor":"top"}]},                        {"responsive": true}                    ).then(function(){
                            Plotly.addFrames('9f8fc860-2069-4249-ba26-2b93fb5e7011', [{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,29.8,null,null,null,null,null,null,null,null,null,29.8,null,null,null,null,null,null,0.6,10.0,null,null,null,74.0,null,null,null,58.8,null,null,null,null,21.0,null,null,23.0,9.8,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,38.4,null],"type":"choropleth"}],"name":"2007-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2008-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","COM","TCD","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,13.1,null,null,null,null,43.0,null,null,null,null,null,null,null,null,null,null,null,null,36.7,null,null,33.2,5.6,null,null,null,27.8,null,null,null,null,56.7,null,null,null,43.0,null],"type":"choropleth"}],"name":"2008-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2009-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,0.9,null,null,null,null,null,null,null,null,null,null,23.2,null,67.2,null,19.7,null,null,null,null,null,null,20.3,null,null,null,8.4,9.1,null,null,null,null,35.8,35.8,null,null,null,59.6,null,23.6],"type":"choropleth"}],"name":"2009-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2010-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,35.1,17.2,null,null,34.1,42.7,null,39.1,null,null,null,null,null,1.5,1.7,null,null,30.2,null,null,51.2,null,null,null,null,null,43.4,34.8,null,null,null,null,null,null,49.1,10.8,null,null,null,62.1,null,null,51.2,65.0,59.1,33.8,null,null,34.0,null],"type":"choropleth"}],"name":"2010-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2011-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,28.3,null,null,null,null,null,23.1,null,null,null,null,null,null,null,null,33.2,null,null,3.6,null,null,52.6,null,null,null,null,57.1,null,19.8,null,null,19.7,null,null,29.9,null,null,44.6,null,null,8.2,null,null,null,null,null,null,null,null,null,64.5,null,2.3],"type":"choropleth"}],"name":"2011-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2012-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,38.4,null,null,25.4,null,null,null,null,26.7,null,25.0,17.5,null,null,null,null,null,null,25.9,null,null,28.1,null,null,null,null,null,null,32.5,null,null,null,null,null,null,19.2,null,null,null,null,null,null,null,null,null,null,53.7,null,null,null,36.9,null],"type":"choropleth"}],"name":"2012-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2013-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,6.7,null,null,null,null,null,55.7,null,11.2,null,22.5,null,null,null,null,8.4,null,32.7,12.0,null,6.2,null,48.3,null,null,31.9,null,null,null,null,null,null,null],"type":"choropleth"}],"name":"2013-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2014-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,25.9,null,49.2,null,null,38.2,null,null,null,29.2,null,null,null,null,null,null,null,null,null,null,48.5,null,28.0,27.0,null,null,null,null,42.4,null,null,null,null,null,null,null,27.3,null,1.4,6.7,null,null,null,null,null,null,null,18.3,null,null,39.8,3.0],"type":"choropleth"}],"name":"2014-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2015-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,26.9,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,27.1,null,null,null,null,null,28.7,null,null,null,38.4,null,null,41.2,11.4,null,3.4,null,null,null,null,null,null,null,null,null,76.9,null,1.0],"type":"choropleth"}],"name":"2015-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2016-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,18.1,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,7.7,null,null,50.1,null,null,null,null,65.5,null,10.1,37.6,null,null,null,null,null,null,null,null,null,null,1.7,null,57.0,null,0.5,null,null,51.1,null,null,71.5,null,null],"type":"choropleth"}],"name":"2016-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2017-01-01<br>Country Code=%{location}<br>Children with fever receiving antimalarial drugs (% of children under age 5 with fever)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,47.0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,29.4,null,null,null,null,null,null,null,null,19.6,null,4.7,null,null,null,null,null,null,36.2,31.1,null,null,null,null],"type":"choropleth"}],"name":"2017-01-01"}]);
                        }).then(function(){

var gd = document.getElementById('9f8fc860-2069-4249-ba26-2b93fb5e7011');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
#Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women) 
fig5 = px.choropleth(Malaria_data,locations=Malaria_data['Country Code'],color=Malaria_data['Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)'],color_continuous_scale='Blues',locationmode='ISO-3',scope='africa',animation_frame=Malaria_data['Year'],title="Malaria in Africa: Intermittent Preventive Treatment of Malaria in Pregnancy",labels={'color':'Intermittent preventive treatment (IPT) of malaria in pregnancy'})

fig5.show()
```


<div>                            <div id="d068326f-24b5-469b-ad38-a99eebb93c03" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("d068326f-24b5-469b-ad38-a99eebb93c03")) {                    Plotly.newPlot(                        "d068326f-24b5-469b-ad38-a99eebb93c03",                        [{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,1.5,null,null,null,null,null,null,null,null,null,2.5,null,null,null,null,null,null,0.5,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,7.0,null,2.0,null,null,null,null,null,null,null,null,null,null,null,null,null,43.1,null],"type":"choropleth"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"geo":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"center":{},"scope":"africa"},"coloraxis":{"colorbar":{"title":{"text":"Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)"}},"colorscale":[[0.0,"rgb(247,251,255)"],[0.125,"rgb(222,235,247)"],[0.25,"rgb(198,219,239)"],[0.375,"rgb(158,202,225)"],[0.5,"rgb(107,174,214)"],[0.625,"rgb(66,146,198)"],[0.75,"rgb(33,113,181)"],[0.875,"rgb(8,81,156)"],[1.0,"rgb(8,48,107)"]]},"legend":{"tracegroupgap":0},"title":{"text":"Malaria in Africa: Intermittent Preventive Treatment of Malaria in Pregnancy"},"updatemenus":[{"buttons":[{"args":[null,{"frame":{"duration":500,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":500,"easing":"linear"}}],"label":"&#9654;","method":"animate"},{"args":[[null],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"&#9724;","method":"animate"}],"direction":"left","pad":{"r":10,"t":70},"showactive":false,"type":"buttons","x":0.1,"xanchor":"right","y":0,"yanchor":"top"}],"sliders":[{"active":0,"currentvalue":{"prefix":"Year="},"len":0.9,"pad":{"b":10,"t":60},"steps":[{"args":[["2007-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2007-01-01","method":"animate"},{"args":[["2008-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2008-01-01","method":"animate"},{"args":[["2009-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2009-01-01","method":"animate"},{"args":[["2010-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2010-01-01","method":"animate"},{"args":[["2011-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2011-01-01","method":"animate"},{"args":[["2012-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2012-01-01","method":"animate"},{"args":[["2013-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2013-01-01","method":"animate"},{"args":[["2014-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2014-01-01","method":"animate"},{"args":[["2015-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2015-01-01","method":"animate"},{"args":[["2016-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2016-01-01","method":"animate"},{"args":[["2017-01-01"],{"frame":{"duration":0,"redraw":true},"mode":"immediate","fromcurrent":true,"transition":{"duration":0,"easing":"linear"}}],"label":"2017-01-01","method":"animate"}],"x":0.1,"xanchor":"left","y":0,"yanchor":"top"}]},                        {"responsive": true}                    ).then(function(){
                            Plotly.addFrames('d068326f-24b5-469b-ad38-a99eebb93c03', [{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2007-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,1.5,null,null,null,null,null,null,null,null,null,2.5,null,null,null,null,null,null,0.5,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,7.0,null,2.0,null,null,null,null,null,null,null,null,null,null,null,null,null,43.1,null],"type":"choropleth"}],"name":"2007-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2008-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","COM","TCD","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,28.0,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,3.2,3.7,null,null,null,5.2,null,null,null,null,7.0,null,null,null,null,null],"type":"choropleth"}],"name":"2008-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2009-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,6.9,null,11.0,null,1.9,null,null,null,null,null,null,null,null,null,null,5.0,15.4,null,null,null,null,null,null,null,null,null,17.0,null,7.0],"type":"choropleth"}],"name":"2009-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2010-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,5.2,0.3,null,null,13.58270713,6.063132461,null,5.442918583,null,null,null,null,null,null,1.117003718,null,null,null,null,null,null,null,null,null,null,null,18.2,null,null,null,null,null,null,null,5.7,null,null,null,null,17.54263277,null,null,4.755756787,0.438459589,2.7,17.48799174,null,null,null,null],"type":"choropleth"}],"name":"2010-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2011-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,8.4,null,null,null,null,null,12.0,null,null,null,null,null,null,null,null,24.3,null,null,null,null,null,39.38870369,null,null,null,null,26.5,null,4.8,null,null,6.46138589,null,null,9.5,null,null,5.953138182,null,null,13.2,null,null,null,null,null,null,null,null,null,10.3,null,5.0],"type":"choropleth"}],"name":"2011-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2012-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,10.0,null,null,0.0,null,null,null,null,11.0,null,12.1,8.1,null,null,null,null,null,null,6.8,null,null,11.2,null,null,null,null,null,null,12.7,10.76045256,null,null,null,null,null,9.1,null,null,null,null,null,null,null,null,null,null,4.0,null,null,null,52.4,null],"type":"choropleth"}],"name":"2012-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2013-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,5.7,null,null,null,null,null,17.9,null,3.9,null,11.8,null,null,null,null,3.4,null,7.1,null,null,4.6,null,20.7,null,null,null,null,null,null,null,null,null,null],"type":"choropleth"}],"name":"2013-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2014-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,12.5,null,22.4,null,null,26.0,null,null,null,5.7,null,null,null,null,null,null,null,null,null,null,38.6,null,18.6,10.3,null,null,null,null,12.6,null,null,null,null,null,null,null,null,null,12.3,2.8,null,null,null,null,null,null,null,22.8,null,null,51.4,5.985314305],"type":"choropleth"}],"name":"2014-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2015-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,19.0,null,null,null,null,null,null,null,8.5,null,null,12.2,null,null,null,null,null,null,null,null,null,null,null,null,22.9,null,null,null,null,null,21.0,11.2,null,null,23.3,null,null,21.4,null,null,11.2,null,null,null,null,null,null,null,null,null,27.5,null,null],"type":"choropleth"}],"name":"2015-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2016-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,20.0,null,null,null,12.6,null,null,null,null,null,null,null,22.6,null,null,null,null,null,null,null,null,59.6,29.9,null,null,null,23.1,null,10.6,30.4,null,null,null,null,null,null,null,null,null,null,22.1,null,31.1,null,null,null,null,8.0,null,null,17.2,null,null],"type":"choropleth"}],"name":"2016-01-01"},{"data":[{"coloraxis":"coloraxis","geo":"geo","hovertemplate":"Year=2017-01-01<br>Country Code=%{location}<br>Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)=%{z}<extra></extra>","locationmode":"ISO-3","locations":["DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COD","COG","CIV","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN","SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"],"name":"","z":[null,null,null,null,null,12.9,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,43.4,null,null,null,null,null,null,null,null,41.1,null,null,null,null,null,null,null,14.9,null,null,22.0,null,26.8,null,null,null,null,25.8,41.7,null,null,null,null],"type":"choropleth"}],"name":"2017-01-01"}]);
                        }).then(function(){

var gd = document.getElementById('d068326f-24b5-469b-ad38-a99eebb93c03');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



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
    

