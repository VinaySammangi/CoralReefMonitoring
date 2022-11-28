#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:18:07 2022

@author: vinaysammangi
"""

import streamlit as st
import json
from io import StringIO
from PIL import Image
import ee
import geemap
import pandas as pd
import geopandas as gpd
import plotly.express as px
from st_aggrid import AgGrid, GridUpdateMode, JsCode, GridOptionsBuilder, ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from datetime import date
from datetime import datetime
import eemont
from joblib import Parallel, delayed
import subprocess
import sys
import plotly.graph_objects as go
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import base64


streamlit_table_theme = "balham"
# ValueError: dark is not valid. Available options: {'STREAMLIT': <AgGridTheme.STREAMLIT: 'streamlit'>, 'ALPINE': <AgGridTheme.ALPINE: 'alpine'>, 'BALHAM': <AgGridTheme.BALHAM: 'balham'>, 'MATERIAL': <AgGridTheme.MATERIAL: 'material'>}
# Landsat8 bands and their descriptions
l8_bands_dict = {"SR_B1":"ULTRABLUE","SR_B2":"BLUE","SR_B3":"GREEN","SR_B4":"RED","SR_B5":"NIR","SR_B6":"SIR1","SR_B7":"SIR2",
                 "SR_QA_AEROSOL":"AEROSOL_ATTRIBUTES", "ST_B10":"SURFACE_TEMPERATURE","ST_ATRAN":"ATMOSPHERIC_TRANSMITTANCE",
                 "ST_CDIST":"PIXEL_DISTANCE","ST_DRAD":"DOWNWELLED_RADIANCE","ST_EMIS":"BAND10_EMISSIVITY",
                 "ST_EMSD":"EMISSIVITY_STD", "ST_QA":"SURFACE_TEMPERATURE_UNCERTAINTY", "ST_TRAD":"RADIANCE_THERMALBAND",
                 "ST_URAD":"UPWELLED_RADIANCE", "QA_PIXEL":"CLOUD", "QA_RADSAT":"RADIOMETRIC_SATURATION"}

# Landsat8 spectral indices available in eemont library
spectral_indices = ["AFRI1600","AFRI2100","ARVI","ATSAVI","AVI","AWEInsh","AWEIsh","BAI","BAIM","BCC",
                    "BI","BLFEI","BNDVI","BWDRVI","BaI","CIG","CSI","CSIT","CVI","DBI","DBSI","DVI","DVIplus",
                    "EBBI","EMBI","EVI","EVI2","ExG","ExGR","ExR","FCVI","GARI","GBNDVI","GCC","GDVI","GEMI",
                    "GLI","GNDVI","GOSAVI","GRNDVI","GRVI","GSAVI","GVMI","IAVI","IBI","IKAW","IPVI","LSWI",
                    "MBI","MBWI","MCARI1","MCARI2","MGRVI","MIRBI","MLSWI26","MLSWI27","MNDVI","MNDWI","MNLI",
                    "MRBVI","MSAVI","MSI","MSR","MTVI1","MTVI2","MuWIR","NBLI","NBR","NBR2","NBRT1","NBRT2",
                    "NBRT3","NBSIMS","NBUI","NDBI","NDBaI","NDDI","NDGI","NDGlaI","NDII","NDISIb","NDISIg",
                    "NDISImndwi","NDISIndwi","NDISIr","NDMI","NDPI","NDSI","NDSII","NDSInw","NDSaII",
                    "NDVI","NDVIMNDWI","NDVIT","NDWI","NDWIns","NDYI","NGRDI","NIRv","NIRvH2","NLI",
                    "NMDI","NRFIg","NRFIr","NSDS","NWI","NormG","NormNIR","NormR","OCVI","OSAVI","PISI","RCC",
                    "RDVI","RGBVI","RGRI","RI","S3","SARVI","SAVI","SAVI2","SAVIT","SI","SIPI","SR","SR2","SWI",
                    "SWM","TDVI","TGI","TSAVI","TVI","TriVI","UI","VARI","VI6T","VIG","VgNIRBI","VrNIRBI","WDRVI",
                    "WDVI","WI1","WI2","WRI","kEVI","kIPVI","kNDVI","kRVI","kVARI"]
spectral_bands = list(l8_bands_dict.keys())
l8_dates = {"start_date":"2013-04-01","end_date":"2022-08-01"}

# Refer to https://medium.com/@mykolakozyr/using-google-earth-engine-in-a-streamlit-app-62c729793007
# for getting the json_data
# Data from the downloaded JSON file
json_data = '''
{
  "type": "service_account",
  "project_id": "practicum-project-363013",
  "private_key_id": "9ed5f5c2a9d6f5ab4f495fa438e5a7f9e972fd52",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC2WTXfYD/ezdP1\n1+P6n+dvBkFVptsHv/3KaKbaXBJRtMiItPndzwgdsgjAU8T1VqU+CJc38FWb7G9a\nTWFn8V2VWIJbuUzgEUNyjoMkfIRx3WxmyceqyMOMQxbM9G96MUp5gmUE8lno+o/N\nvFmz9O+Gua48xbR89E5PJgyl+jLEmDr+7OYj/srh0Ja18HvtqrkJHAdoHlNS6CdM\nA7Da+DDPPQfuOM8HOB8ZoTUgewPcXYxC9hbmzQl81NfUibu6NbA6IiAfGcDM5bfh\nrQMwpKbbQqye361NVg6PW91mWBXTnV+xNd0MFXsVygv89kcZ2EnTy/77KOcXnu1v\n5hv8mCjnAgMBAAECggEALRTYI04N7F0Rsp150Qv4cTPoMi9Kxls6ePCvk5ugsc+S\npm2ruqFFHeZWkIoFTyxpNPF1xVAnMiHdk8M+ui5rlxEnRVsF/P13odpG5N3d9rKp\n6q2nLfttkP9DI0+pQdnu0iShKfxqqxVLOS+AM+Px1eqQ/5hXW28g7yN2jBBTvdOC\n+hmHBGx5xoM+x7/SstHMtUX14MTqmcgWzslvUBFyvxUs07ksQuLjnpmz4V3F9NCF\nBOe/L+0BhtDMCeZ8WY0Up4ip/d039Kj/h4W/ErFJGI4pqE+69iC2A+sauvQbpzif\ngywQsPK/F1dg5UFhh4yr9qRCUgO90rkPinrF63uSkQKBgQD6tspwBW9DM9h7fssu\n9fYK9dgiR28oEBNBJ3etNLUIj1rQKlURaolwvAcX9IQwWZKeWNZuib5nhEAxyBQW\nNx607QdEaM7B1OSPQMy8i2SCizRGbJM8m+gsjg/RxyVCcT83zokSWB+1DgpFPuUa\nAhhCM7t8rCA+N9SiujesIGis8QKBgQC6MWwIS0xumOI5/nQcrjI3/BN1KDlfuNC7\n+XTXiSWLAdSNXxHSW0aXJcySV/sKK3NUJltpc/TM5wugCqJfWfimdguwl+X1JqWe\nRGjx2QOBMhprb0HDUlN8kyEjWD4/WTsjpXjqIQdVrMPfaX45NGBoZT7PxZ63EjGy\n6QbK1SeTVwKBgCniwf1nGwiKL9+p9j4ZP4rjOcG4V3zE+sKG2nqodJpCgPSILgAj\n4WRhNXouEquVO2aTBvgesR3QPX1TpO91M/8cHnuyWuCNNcYtGEdjrl4U7Z3aY9rb\nXTWcYk40zCfGjb5AFixnZpy0BMk+0b2/ndfplqgkhZp/b1nkbIqoO3SxAoGBAKMW\nPwZUzjHhf+ZEVvf4LMyU44YvIXISs+KycgGIg3XquH7L0xRqFr61wSY+IgmaXX5L\nyq3nf3kqtygLqIXUjNNheoPHyQiePVsPmMydxVAYzsNjxDqNlcr8JH6NAJkEU6S5\nf9uz6nTEyxyZjpIUqo1GgWoEMy0vppCLRAPOCMgpAoGBAM9yTqMfxpo3K5hWqbI8\naL7prFVK3EVlhRqBrpnjfnqBfN0XEGp6oVCPOhXvVWIF4w4l4ASFyEuivqmICvqd\ngE1ZRvLo1s/lYIKL96W2n/aiwehA0c//Ahb+ongl3dcugnr13xxPKzBO5YRnx07s\nhZ+Q1i4L1xTIF9oqngNzB1Q+\n-----END PRIVATE KEY-----\n",
  "client_email": "earth-engine-resource-viewer@practicum-project-363013.iam.gserviceaccount.com",
  "client_id": "110078860645894310593",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/earth-engine-resource-viewer%40practicum-project-363013.iam.gserviceaccount.com"
}
'''

# Preparing values
json_object = json.loads(json_data, strict=False)
service_account = json_object['client_email']
json_object = json.dumps(json_object)
# Authorising the app
credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
ee.Initialize(credentials)

BACKGROUND_COLOR = 'white'
COLOR = 'black'

def v_spacer(height,elem):
    for _ in range(height):
        elem.write('\n')
        
def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = True,
        padding_top: int = 0, padding_right: int = 0.5, padding_left: int = 0.5, padding_bottom: int = 0,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        st.markdown(
            f'''
            <style>
                .reportview-container .sidebar-content {{
                    padding-top: {padding_top}rem;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

im = Image.open("imgs/gt_logo.png")
st.set_page_config(page_title="GOTECH - Reef Monitoring",page_icon=im,layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
                .css-18e3th9 {
                    padding-top: 1rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)


st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
 
file_ = open("imgs/nasa_logo.png", "rb")
contents = file_.read()
nasa_url = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("imgs/coralvita_logo1.png", "rb")
contents = file_.read()
coralvita_url = base64.b64encode(contents).decode("utf-8")
file_.close()

file_ = open("imgs/gt_logo.png", "rb")
contents = file_.read()
gt_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown("<p style='text-align: right; color:orange;'><i>Please view this website on either desktop or laptop</i></p>",unsafe_allow_html=True)
st.markdown(f"<h1 width: fit-content; style='text-align: center; color: white; background-color:#0083b8;'>Coral Reef Monitoring System <img src='data:image/png;base64,{coralvita_url}' alt='Coral Vita' align='right' height='55'> <img src='data:image/png;base64,{nasa_url}' alt='NASA' align='right' height='60'> <img src='data:image/png;base64,{gt_url}' alt='GaTech' align='right' height='55'> </h1>", unsafe_allow_html=True)        
set_page_container_style()

@st.cache(allow_output_mutation=True)
def _load_data():
    data_dict = {}
    data_dict["coral_df_keys"] = pd.read_pickle("Input/coral_df_keys.pkl")
    presence_df = pd.read_pickle('Input/ModelData.pkl')
    presence_gdf = gpd.GeoDataFrame(presence_df, geometry='geometry')
    presence_gdf["LONG"] = presence_gdf.geometry.centroid.x
    presence_gdf["LAT"] = presence_gdf.geometry.centroid.y
    presence_gdf["date"] = pd.to_datetime(presence_gdf["date"])
    presence_gdf["year"] = presence_gdf["date"].dt.year
    presence_gdf["month"] = presence_gdf["date"].dt.month
    presence_gdf["hour"] = presence_gdf["date"].dt.hour
    data_dict["presence_gdf"] = presence_gdf
    data_dict["boundaries_regions"] = gpd.read_file('Input/boundaries_regions.geojson')
    data_dict["presence_model"] = joblib.load("Output/presence_models_with_ReefRegion.pkl")
    data_dict["bleaching_model"] = joblib.load("Output/bleaching_models_with_ReefRegion.pkl")
    data_dict["Presence_Summary"] = pd.read_excel("Input/ModelSummaries.xlsx",sheet_name="Presence")
    data_dict["Binary_Summary"] = pd.read_excel("Input/ModelSummaries.xlsx",sheet_name="Bleaching-binary",header=0)
    data_dict["Multi_Summary"] = pd.read_excel("Input/ModelSummaries.xlsx",sheet_name="Bleaching-multi",header=0)
    return data_dict

def _data_exploration():
    tab1_1, tab1_2, tab1_3 = st.tabs(["1.1. Coral Databases","1.2. Satellite Instruments","1.3. Integration"])
    with tab1_1:
        # st.write("**Technical Area 1:** Cross-Validate the Open-Source Reef Databases")
        tab1_1_1, tab1_1_2, tab1_1_3, tab1_1_4 = st.tabs(["1.1.1. Allen Coral Atlas","1.1.2. NOAA NCEI","1.1.3. GCBD","1.1.4. AIMS"])
        with tab1_1_1:
            _allencoral()
        with tab1_1_2:
            _coral_noaa()
        with tab1_1_3:
            _gcbd()
        with tab1_1_4:
            _aims()

    with tab1_2:
        # st.write("**Technical Area 2:** Time-Align and Geo-Align with Corresponding Instrument Data")
        tab1_2_1, = st.tabs(["1.2.1. LANDSAT"])
        with tab1_2_1:
            _landsat_description()

    with tab1_3:
        pass
        _data_integration()


def _data_integration():
    st.markdown("After combining & processing the data across NOAA NCEI, GCBD, AIMS databases from April 2013, we ended up with 11,510 coral points. \
                We were able to download the Landsat8 data for 9,303 points (81%). The data processing codes can be found at `1. Coral Databases/1.1. Coral/\
                Scripts/Coral Data Processing.ipynb` & `2. Landsat8/Scripts/EE_Landsat8.ipynb`. We then clustered the data points into 9 geographical \
                regions, which is shown in the table below. We can notice that most of the data is concentrated around \
                florida, bahamas, and great barrier reef regions. We then defined clusters based on the closest regions \
                and the number of data points in each cluster. Cluster 1 covers 5 regions with 6,958 coral points,\
                Cluster 2 covers 4 regions with 4,552 coral points.")
    
    data_file = _load_data()
    coral_df_keys = data_file["coral_df_keys"].copy()
    boundaries_regions = data_file["boundaries_regions"].copy()
    presence_gdf = data_file["presence_gdf"].copy()
    presence_gdf = presence_gdf.rename(columns={"Coral_Class":"Class"})
    regions = list(boundaries_regions["name"])
    
    counts_df = coral_df_keys.groupby("ReefRegion").agg({"ID":"count"}).reset_index()
    counts_df["Cluster"] = counts_df["ReefRegion"].map({"Northern Caribbean - Florida,Bahamas":"Cluster1","Great Barrier Reef and Torres Strait":"Cluster2",
      "Southeastern Caribbean":"Cluster1","Mesoamerica":"Cluster1",
      "Central South Pacific":"Cluster2","Subtropical Eastern Australia":"Cluster2",
      "Coral Sea":"Cluster2","Bermuda":"Cluster1",
      "Eastern Tropical Pacific":"Cluster1"})
    counts_df.columns = ["Region","Coral Count","Cluster"]
    counts_df = counts_df.sort_values("Coral Count",ascending=False)
    st1_31, st1_32, st1_33 = st.columns([2,3,2],gap="medium")
    st1_32.markdown("<h4 width: fit-content; style='text-align: center; color: gray;'>Region-Wise Coral Count</h4>", unsafe_allow_html=True)
    gb = GridOptionsBuilder.from_dataframe(counts_df)
    gb.configure_grid_options(enableCellTextSelection=True)
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True,value=True,enableRowGroup=True,aggFunc="sum",editable=False)
    gridOptions = gb.build()

    with st1_32.container():
        AgGrid(counts_df, gridOptions=gridOptions,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                theme=streamlit_table_theme,
                enable_enterprise_modules=True)
    st1_32.markdown("<p style='color:#0083b8;'> <i> please refresh the page if the table is not visible properly. </i> </p>",unsafe_allow_html=True)
    st.markdown("Then, we also sampled the non coral data points from Allen Coral database with equal proportions \
                taken from coral counts in each of the 9 regions. The data processing codes can be found at `1. Non Coral Databases/1.1. Coral/\
                Scripts/NonCoral Data Processing.ipynb` & `2. Landsat8/Scripts/EE_Landsat8.ipynb`. The following figure shows the region wise coral, \
                non coral data points plotted on the map.")
    
    selected_regions = st.multiselect('Regions to consider',regions,default=regions)
    presence_gdf_filtered = presence_gdf.loc[(presence_gdf["ReefRegion"].isin(selected_regions)),]
    boundaries_regions_filtered = boundaries_regions.loc[boundaries_regions["name"].isin(selected_regions)].reset_index(drop=True)
    layers_ = []
    for i in range(boundaries_regions_filtered.shape[0]):
      main_dict = {}
      temp = boundaries_regions_filtered.iloc[i:(i+1)]
      main_dict["source"] = json.loads(temp.geometry.to_json())
      main_dict["type"] = "line"
      main_dict["below"] = "traces"
      main_dict["color"] = list(temp["color"])[0]
      layers_.append(main_dict)

    fig = px.scatter_mapbox(presence_gdf_filtered, lat="LAT", lon="LONG", hover_data=["ReefRegion","year", "month","hour"],color = "Class",
                            color_discrete_map={"Coral":"green","NonCoral":"red"}, opacity=0.2)
    fig.update_layout(
        title_text="Region-Wise Coral & NonCoral Data",title_x=0.5,title_y=0.95,
        mapbox = {
            'style': "open-street-map",
            'center': { 'lon': -130, 'lat':-8},
            'zoom': 2, 'layers': layers_},
        legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        )),
        margin = {'l':0, 'r':0, 'b':0, 't':50},height=500
        )
    st.plotly_chart(fig,use_container_width=True)
    

def _landsat_description():
    st.markdown("Landsat 8 collects the data in three shortwave infrared bands and two thermal infrared bands, in \
    addition to near infrared (VNIR). The sensors aboard Landsat 8 were designed to have a higher \
    sensitivity to brightness and color than their predecessors. This extra sensitivity has made it easier \
    to spot coral reefs and quantify their area and depth. The data is freely available to download in \
    different forms.")
    pass

def _coral_noaa():
    st.markdown("NCEI is a global coral bleaching database consisting of 33,244 records of the presence or \
                absence of coral bleaching from 1963-2021, compiled from sources such as Coral Reef Watch \
                (CRW) and Donner database. The dataset comprises columns such as percentage bleaching and \
                percentage mortality which are essential features in estimating coral vitality through \
                satellite imagery. The database also includes a source column which would be useful for \
                removing any potential overlap with the other global bleaching databases.")
    st.markdown("<a href='https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0228498'>Download URL</a>",unsafe_allow_html=True)
    st.markdown("Data Processing can be found at `1. Coral Databases/1.1. Coral/Scripts/Coral Data Processing.ipynb`")

def _gcbd():
    st.markdown("Global Coral-Bleaching Database (GCBD) encompasses 34,846 coral bleaching records from 14,405 \
                sites in 93 countries from 1980-2020. The GCBD provides vital information on the presence or \
                absence of coral bleaching, site exposure, distance to land, mean turbidity, cyclone frequency,\
                and a suite of sea-surface temperature metrics during the survey. The Global Coral Bleaching \
                Database (GCBD) is an SQLite file containing 20 related tables.")
    st.markdown("<a href='https://springernature.figshare.com/articles/dataset/Global_Coral_Bleaching_Database/17076290?backTo=/collections/A_Global_Coral-Bleaching_Database_GCBD_1998_2020/5314466'>Download URL</a>",unsafe_allow_html=True)
    st.markdown("Data Processing can be found at `1. Coral Databases/1.1. Coral/Scripts/Coral Data Processing.ipynb`")

def _aims():
    st.markdown("AIMS Long-Term Reef Monitoring Program monitors coral presence across 64 key reefs and 11 \
                sectors of the Great Barrier Reef region through manta tow survey results. The dataset has \
                2,452 records (from 1992 to 2021), including estimates of the percentage cover of living \
                hard corals, soft coral, and recently dead hard coral.")
    st.markdown("<a href='https://apps.aims.gov.au/metadata/download/10.25845/5c09b0abf315a/manta-tow-by-reef.zip'>Download URL</a>",unsafe_allow_html=True)
    st.markdown("Data Processing can be found at `1. Coral Databases/1.1. Coral/Scripts/Coral Data Processing.ipynb`")

def _allencoral():
    st.markdown("The Allen Coral Atlas is a large open-source data source that maps the world's coral reefs \
                and monitors their threats to provide actionable data and a shared understanding of coastal \
                ecosystems. It has data on the presence of Coral/Algae, Seagrass, Rock, Rubble, Sand, \
                Microalgal Mats, etc. Since this database doesn't differentiate between Coral and \
                Algae, we will primarily gather noncoral data from here.")
    st.markdown("<a href='https://allencoralatlas.org/atlas/'>Download URL</a>",unsafe_allow_html=True)
    st.markdown("We came to know during our project that coral data points collected in some regions are \
                more reliable. So, once you visit the URL, you can download the latest mapped data in these\
                regions (listed below) from the Mapped Areas tab.")
    st.markdown("<h4>Regions</h4>",unsafe_allow_html=True)                
    st.markdown("- Bermuda")
    st.markdown("- Coral Sea")
    st.markdown("- Eastern Tropical Pacific")
    st.markdown("- Subtropical Eastern Australia")
    st.markdown("- Central South Pacific")
    st.markdown("- Mesoamerica")
    st.markdown("- Southeastern Caribbean")
    st.markdown("- Great Barrier Reef and Torres Strait")
    st.markdown("- Northern Caribbean - Florida,Bahamas")    
    st.markdown("Data Processing can be found at `1. Coral Databases/1.2. Non Coral/Scripts/NonCoral Data Processing.ipynb`")

def _model_results():
    tab2_1, tab2_2 = st.tabs(["2.1. Coral Presence","2.2. Bleaching"])
    with tab2_1:
        _presence_experiments()
    with tab2_2:
        _bleaching_experiments()

def _bleaching_experiments():
    st.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'>Binary Classifier vs Regressor Results : Summary</h3>", unsafe_allow_html=True)
    data_file = _load_data()
    model_summaries = data_file["Binary_Summary"].copy()
    gb = GridOptionsBuilder.from_dataframe(model_summaries)
    gb.configure_grid_options(enableCellTextSelection=True)
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True,value=True,enableRowGroup=True,aggFunc="sum",editable=False)
    gridOptions = gb.build()
    
    with st.container():
        AgGrid(model_summaries, gridOptions=gridOptions,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                theme=streamlit_table_theme,
                enable_enterprise_modules=True)
        
    st.markdown("<p style='color:#0083b8;'> <i> please refresh the page if the table is not visible properly. </i> </p>",unsafe_allow_html=True)        
    st.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'>Multi-label Classifier vs Regressor Results : Summary</h3>", unsafe_allow_html=True)
    model_summaries = data_file["Multi_Summary"].copy()
    gb = GridOptionsBuilder.from_dataframe(model_summaries)
    gb.configure_grid_options(enableCellTextSelection=True)
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True,value=True,enableRowGroup=True,aggFunc="sum",editable=False)
    gridOptions = gb.build()
    
    with st.container():
        AgGrid(model_summaries, gridOptions=gridOptions,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                theme=streamlit_table_theme,
                enable_enterprise_modules=True)
    st.markdown("<p style='color:#0083b8;'> <i> please refresh the page if the table is not visible properly. </i> </p>",unsafe_allow_html=True)        
    st.markdown("We observe that for both balanced, imbalanced thresholds, binary & multi-label classifier outperforms \
                the regressor model. The confusion matrix and the feature importance plot for the second binary classifier model (imbalanced) are also provided below.\
                The codes are available at `3. Modeling/Scripts/3. Bleaching - Data Preparation.ipynb` and `4. \
                Modeling/Scripts/4. Bleaching - Training.ipynb`.")
                
    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'> Confusion Matrix & Feature Importance</h4>", unsafe_allow_html=True)
    image1 = Image.open('imgs/bleaching_cm.png')
    image2 = Image.open('imgs/bleaching_fi.png')
    image1 = image1.resize((600, 600))
    image2 = image2.resize((900, 600))
    st2_1_1, st2_1_2 = st.columns([600,900],gap="small")
    st2_1_1.image(image1, caption='Classification Report')
    st2_1_2.image(image2, caption='Feature Importance')
    st2_1_2.markdown("From the Feature Importance plot, we observe that all the important variables except `Southeastern Caribbean`, `SR_B7` (shortwave infrared 2 band),\
                     `SR_B6` (shortwave infrared 1 band), `SR_B1` (ultra blue - coastal aerosol band)\
                     are spectral indices. Please refer to section 1.2.1. for understanding more about these indices.")

    st2_1_1.markdown("From the confusion matrix, we observe that the model predicts Low bleaching better than High bleaching. \
                     Now, we try to understand more about the prediction probabilites.")
    
    # st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'> Prediction Probabilities Analysis</h4>", unsafe_allow_html=True)
    
    
def _presence_experiments():
    st.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'>Model Results : Summary</h3>", unsafe_allow_html=True)
    data_file = _load_data()
    model_summaries = data_file["Presence_Summary"].copy()
    gb = GridOptionsBuilder.from_dataframe(model_summaries)
    gb.configure_grid_options(enableCellTextSelection=True)
    # gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True,value=True,enableRowGroup=True,aggFunc="sum",editable=False)
    gridOptions = gb.build()
    
    with st.container():
        AgGrid(model_summaries, gridOptions=gridOptions,
                columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                theme=streamlit_table_theme,
                enable_enterprise_modules=True)
    st.markdown("<p style='color:#0083b8;'> <i> please refresh the page if the table is not visible properly. </i> </p>",unsafe_allow_html=True)
    st.markdown('**Cluster1 Regions:** Northern Caribbean - Florida,Bahamas , Southeastern Caribbean , Mesoamerica , Bermuda , Eastern Tropical Pacific')
    st.markdown('**Cluster2 Regions:** Great Barrier Reef and Torres Strait , Central South Pacific , Subtropical Eastern Australia , Coral Sea')
    
    st.markdown("<h3 width: fit-content; style='text-align: center; color: gray;'> Best Model Results : Details</h3>", unsafe_allow_html=True)

    # st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'>1. Model 1</h4>", unsafe_allow_html=True)
    st.markdown('We can observe that the best model is trained and predicted on the global dataset using Stratified 5-Fold \
                Cross Validation & including `Reef Region` as a dummy variable. The confusion matrix and the feature \
                importance plot are also provided below. The codes are available at `3. Modeling/Scripts/1. Presence - Data \
                Preparation.ipynb` and `3. Modeling/Scripts/2. Presence - Training.ipynb`.')
    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'> Confusion Matrix & Feature Importance</h4>", unsafe_allow_html=True)
    image1 = Image.open('imgs/presence_cm.png')
    image2 = Image.open('imgs/presence_fi.png')
    image1 = image1.resize((600, 600))
    image2 = image2.resize((900, 600))
    st2_1_1, st2_1_2 = st.columns([600,900],gap="small")
    st2_1_1.image(image1, caption='Classification Report')
    st2_1_2.image(image2, caption='Feature Importance')
    st2_1_2.markdown("From the Feature Importance plot, we observe that all the important variables except `Great Barrier Reef and Torres Strait`, `SR_B3` (green band)\
                     are spectral indices. Please refer to section 1.2.1. for understanding more about these indices.")

    st2_1_1.markdown("From the confusion matrix, we observe that the model's performance is almost same while \
                     predicting coral & non coral data points. Now, we try to understand more about the prediction probabilites.")
    
    st.markdown("<h4 width: fit-content; style='text-align: left; color: gray;'> Prediction Probabilities Analysis</h4>", unsafe_allow_html=True)
    image3 = Image.open('imgs/c_c.png')
    image4 = Image.open('imgs/c_nc.png')
    image5 = Image.open('imgs/nc_c.png')
    image6 = Image.open('imgs/nc_nc.png')
    image3 = image3.resize((800, 600))
    image4 = image4.resize((800, 600))
    image5 = image5.resize((800, 600))
    image6 = image6.resize((800, 600))
    st2_1_3, st2_1_4, st2_1_5, st2_1_6 = st.columns([1,1,1,1],gap="small")
    st2_1_3.image(image3, caption='Actual: Coral; Prediction: Coral')
    st2_1_4.image(image4, caption='Actual: Coral; Prediction: NonCoral')
    st2_1_5.image(image5, caption='Actual: NonCoral; Prediction: Coral')
    st2_1_6.image(image6, caption='Actual: NonCoral; Prediction: NonCoral')
    st2_1_3.markdown("Most of the rightly classified coral data points have probabilities leaning towards 1")
    st2_1_4.markdown("Most of the misclassified coral data points have probabilities leaning towards 0.5")
    st2_1_5.markdown("Most of the misclassified noncoral data points have probabilities leaning towards 0.5")
    st2_1_6.markdown("Most of the rightly classified noncoral data points have probabilities leaning towards 0")
    st.markdown("<p style='color:#0083b8;'> <i> All the 4 plots above communicate that we are more confident in a prediction if it is close to 0 or 1. We should be uncertain if its close to 0.5. </i> </p>",unsafe_allow_html=True)
    image7 = Image.open('imgs/presence_0_1_9_10.png')
    image8 = Image.open('imgs/presence_1_2_8_9.png')
    image9 = Image.open('imgs/presence_2_3_7_8.png')
    image10 = Image.open('imgs/presence_3_4_6_7.png')
    image11 = Image.open('imgs/presence_4_6.png')
    image7 = image7.resize((600, 600))
    image8 = image8.resize((600, 600))
    image9 = image9.resize((600, 600))
    image10 = image10.resize((600, 600))
    image11 = image11.resize((600, 600))
    st2_1_7, st2_1_8, st2_1_9 = st.columns([1,1,1],gap="small")
    st2_1_7.image(image7, caption='Model Performance with prediction probabilities in [0,0.1) and (0.9,1]. It covers 29.7% of the points.')
    st2_1_8.image(image8, caption='Model Performance with prediction probabilities in [0.1,0.2) and (0.8,0.9]. It covers 22.1% of the points.')
    st2_1_9.image(image9, caption='Model Performance with prediction probabilities in [0.2,0.3) and (0.7,0.8]. It covers 18.2% of the points.')
    st2_1_7.image(image10, caption='Model Performance with prediction probabilities in [0.3,0.4) and (0.6,0.7]. It covers 15.3% of the points.')
    st2_1_8.image(image11, caption='Model Performance with prediction probabilities in [0.4,0.6]. It covers 14.8% of the points.')
    st2_1_9.markdown("<u> <p style='color:#0083b8;'> Findings:\
                    </p> </u>",unsafe_allow_html=True)
    st2_1_9.markdown("<ul> <li> <p style='color:#0083b8;'> <i> Based on these plots, we can conclude that model predictions are more accurate if its close to 0 or 1. </i> </p> </li>\
                     <li> <p style='color:#0083b8;'> <i> The accuracy is pretty high if the prediction probability is either >0.8 or <0.2. These thresholds also covers about 51.8% of the overall points. </i> </p> </li>\
                    </ul>",unsafe_allow_html=True)
    
def _real_time_prediction():
    st3_1_1, st3_1_2, st3_1_3, st3_1_4, st3_1_6 = st.columns([2,2,2,7,1])
    lat_ = st3_1_1.number_input('Latitude', -90.0, 90.0,24.676480,format="%.6f",key="lat_3",help='Select a value between -90 and 90')
    long_ = st3_1_2.number_input('Longitude', -180.0, 180.0,-76.216350,format="%.6f",key="long_3",help='Select a value between -180 and 180')
    radius_ = st3_1_3.number_input('Radius', 30, 1000, 100, key="radius_3",help='Select a value between 30 and 1000 meters',step=10)
    st3_1_4.markdown("<p style='color:#0083b8;'> <i> Please wait for 2-3 mins once you enter latitude, longitude, radius values.</i> </p>",unsafe_allow_html=True)
    data_file = _load_data()
    def start_capture():
        geometry = ee.Geometry.Point([long_, lat_]).buffer(radius_)
        
        if lat_==24.676480 and long_==-76.216350 and radius_==100:
            landsat8_df = pd.read_pickle("Output/landsat8_df_default.pkl")
        else:
            geometry = ee.Geometry.Point([long_, lat_]).buffer(radius_)
            fc = ee.FeatureCollection([ee.Feature(geometry)])
            bands_ = spectral_bands + spectral_indices
            s_date = date.today()
            e_date = datetime.strptime(l8_dates["start_date"], '%Y-%m-%d')
            year_end = (s_date.year)
            year_start = (e_date.year)
            start_dates, end_dates = [l8_dates["start_date"]], ["2013-12-31"]
            for year in range(year_start+1,year_end+1):
                start_dates.append(str(year)+"-01-01")
                end_dates.append(str(year)+"-12-31")
            
            landsat8_list = Parallel(n_jobs=-1)(delayed(download_landsat8)(fc,start_dates[i],
                                                                          end_dates[i],bands_) 
                                                for i in range(len(start_dates)))
            landsat8_df = pd.concat(landsat8_list).reset_index(drop=True)
            landsat8_df["date"] = pd.to_datetime(landsat8_df["date"])
            landsat8_df = landsat8_df.sort_values("date",ascending=True).reset_index(drop=True)
            landsat8_df = landsat8_df.dropna().reset_index(drop=True)
            # landsat8_df.to_pickle("Output/landsat8_df_default.pkl")
        return landsat8_df,geometry
    
    def run_cap():
        cap_button = st3_1_6.button("Predict") # Give button a variable name
        if cap_button: # Make button a condition.

            boundaries_regions = data_file["boundaries_regions"]
            reef_region, reef_region_dict = "",{}
            for ind, row in boundaries_regions.iterrows():
                reef_region_dict[row["name"]] = [0]
                if shape(row["geometry"]).contains(Point(long_,lat_)):
                    reef_region = row["name"]
                    reef_region_dict[row["name"]] = [1]
                    
            if reef_region=="":
                st.error('Provided Lat Long is outside the bounding boxes of 9 regions', icon="🚨")
            else:
                with st.spinner('Wait for it...'):
                    landsat8_df,geometry = start_capture()
                    reef_region_df = pd.DataFrame(reef_region_dict)
                    # print(landsat8_df.shape)
                    reef_region_df = reef_region_df.append([reef_region_df]*(landsat8_df.shape[0]-1),ignore_index=True)
                    # print(reef_region_df.shape)
                    landsat8_df = landsat8_df.reset_index(drop=True).join(reef_region_df.reset_index(drop=True))
                    final_gdf = data_file["presence_gdf"].copy()
                    st3_2_1, st3_2_2 = st.columns([5,2])
        
                    fig = go.Figure(go.Scattermapbox(
                        mode = "markers",
                        lon = [], lat = [],
                        marker = {'size': 0, 'color': ["cyan"]}))
                    temp = (geometry.getInfo()["coordinates"][0])
                    x,y = list(np.mean(np.array(temp[:-1]),axis=0))
                    fig.update_layout(
                        title_text="Bounding Box",title_x=0.5,title_y=0.95,
                        mapbox = {
                            'style': "open-street-map",
                            'center': { 'lon': x, 'lat':y},
                            'zoom': 12, 'layers': [{
                                'source': geometry.getInfo(),
                                'type': "line", 'below': "traces", 'color': "#0042FF"},
                    {'source': json.loads(final_gdf.loc[final_gdf["Coral_Class"]=="Coral",].geometry.to_json()),
                    'type': "fill", 'below': "traces", 'color': "green"},
                    {'source': json.loads(final_gdf.loc[final_gdf["Coral_Class"]=="NonCoral",].geometry.to_json()),
                    'type': "fill", 'below': "traces", 'color': "red"}]},
                            margin = {'l':0, 'r':0, 'b':0, 't':50},height=500
                            )                
                    st3_2_2.plotly_chart(fig,use_container_width=True)
                    
                    presence_model = data_file["presence_model"]
                    x_vars  = list(presence_model.feature_names_in_)               
                    presence_predict_probs = presence_model.predict_proba(landsat8_df[x_vars])              
                    presence_predict_probs = (list(presence_predict_probs[:,0]))
                    presence_predictions = list(presence_model.predict(landsat8_df[x_vars]))
                    # print(presence_predictions)
                    presence_predictions_text = presence_predictions.copy()#["NonCoral" if i==0 else "Coral" for i in predictions]
                    dates = list(landsat8_df["date"])
                    color_map = {"NonCoral": "red","Coral": "green"}
                    fig = go.Figure(data=go.Scatter(x=dates,
                                    y=presence_predict_probs,    
                                    mode='markers',                                
                                    marker = {'color': pd.Series(presence_predictions).apply(lambda x: color_map[x]),
                                      'size': 7
                                      },
                                    hovertemplate =
                        'Date: <b>%{x}</b>'+
                        '<br>Probability: <b>%{y}</b>'+
                        # '<br>Class: <b>%{marker.color}</b><br>'
                        '<br>Class: <b>%{text}</b><br>'+                                
                        '<b>%{customdata}</b>',text = presence_predictions_text,customdata = dates))
                    fig.update_layout(margin=dict(l=0,r=0,b=0,t=50))
                    fig.update_layout(xaxis_title="Date",yaxis_title="Coral Probability",title_text="Coral Presence Over Time",title_x=0.5,title_y=0.95,
                                     width=1200,height=500,template="plotly_white",font=dict(size=16))
                    st3_2_1.plotly_chart(fig,use_container_width=True)
    
                    landsat8_df["Coral Presence"] = presence_predictions
                    if landsat8_df.loc[landsat8_df["Coral Presence"]=="Coral",].shape[0]>0:
                        bleaching_model = data_file["bleaching_model"]
                        x_vars  = list(bleaching_model.feature_names_in_)               
                        predict_probs = bleaching_model.predict_proba(landsat8_df.loc[landsat8_df["Coral Presence"]=="Coral",x_vars])              
                        predict_probs = (list(predict_probs[:,0]))
                        predictions = list(bleaching_model.predict(landsat8_df.loc[landsat8_df["Coral Presence"]=="Coral",x_vars]))
                        # print(predict_probs)
                        # print(predictions)
                        predictions_text = predictions.copy()#["NonCoral" if i==0 else "Coral" for i in predictions]
                        dates = list(landsat8_df.loc[landsat8_df["Coral Presence"]=="Coral","date"])
                        color_map = {"High Bleaching": "red","Low Bleaching": "green"}
                        fig = go.Figure(data=go.Scatter(x=dates,
                                        y=predict_probs,    
                                        mode='markers',                                
                                        marker = {'color': pd.Series(predictions).apply(lambda x: color_map[x]),
                                          'size': 7
                                          },
                                        hovertemplate =
                            'Date: <b>%{x}</b>'+
                            '<br>Probability: <b>%{y}</b>'+
                            # '<br>Class: <b>%{marker.color}</b><br>'
                            '<br>Class: <b>%{text}</b><br>'+                                
                            '<b>%{customdata}</b>',text = predictions_text,customdata = dates))
                        fig.update_layout(margin=dict(l=0,r=0,b=0,t=50))
                        fig.update_layout(xaxis_title="Date",yaxis_title="High Bleaching Probability",title_text="Bleaching (Low: < 25% vs High: >= 25%) Over Time",title_x=0.5,title_y=0.95,
                                         width=1200,height=500,template="plotly_white",font=dict(size=16))
                        st3_2_1.plotly_chart(fig,use_container_width=True)
        
                st3_1_6.success('', icon="✅")
    run_cap()      
    # boundaries_regions = data_file["boundaries_regions"]
    # reef_region, reef_region_dict = "",{}
    # for ind, row in boundaries_regions.iterrows():
    #     reef_region_dict[row["name"]] = [0]
    #     if shape(row["geometry"]).contains(Point(long_,lat_)):
    #         reef_region = row["name"]
    #         reef_region_dict[row["name"]] = [1]
            
    # if reef_region=="":
    #     st.error('Provided Lat Long is outside the bounding boxes of 9 regions', icon="🚨")
    # else:
    #     run_cap(pd.DataFrame(reef_region_dict))

from shapely.geometry import shape, GeometryCollection, Point

def download_landsat8(fc_,start_date,end_date,bands_):
    import eemont
    json_object = json.loads(json_data, strict=False)
    service_account = json_object['client_email']
    json_object = json.dumps(json_object)
    # Authorising the app
    credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
    ee.Initialize(credentials)

    S2 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(fc_)
            .filterDate(start_date,end_date)
            .preprocess()
            .select(spectral_bands)
            .spectralIndices(spectral_indices))

    ts = S2.getTimeSeriesByRegions(reducer = [ee.Reducer.median()],
                                              collection=fc_,
                                              bands = bands_,
                                              scale = 30)
    time_df = geemap.ee_to_pandas(ts)
    time_df_filtered = time_df.loc[time_df["QA_PIXEL"]!=-9999,]
    return time_df_filtered
    
def _future_work():
    st.markdown("<h4>Assumptions</h4>",unsafe_allow_html=True)                
    st.markdown("- Coral exists within 100 meters around a Latitude-Longitude point where coral was found.")
    st.markdown("- Coral levels are relatively unchanged within 1 month around the date of data collection.")
    st.markdown("- There are certain regions which are studied more frequently and this results in more accurate data collections.")
    st.markdown("- If Landsat 8 is ever discontinued, the same methods outlined here can be used with data collected from Landsat 9.")

    st.markdown("<h4>Future Work</h4>",unsafe_allow_html=True)                
    st.markdown("- Identify the right thresholds for the bleaching model.")
    st.markdown("- Further improve the models through stacking, ensemble methods, and further hyperparameter tuning")
    st.markdown("- Utilize more coral datasets.")
    st.markdown("- Incorporate region specific factors.")
    st.markdown("- Replicate the results with USGS API instead of EarthEngine.")
    st.markdown("- Explore other satellites like Sentinel.")    
    st.markdown("- Perform feature engineering on spectral bands.")        
    pass
    
tab1, tab2, tab3, tab4 = st.tabs(["1. Datasets","2. Model Results","3. Real-time Prediction","4. Future Work"])

with tab1:
    _data_exploration()
    pass

with tab2:
    _model_results()
    pass

with tab3:
    _real_time_prediction()
    pass

with tab4:
    _future_work()
    pass
