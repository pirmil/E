"""
Import the data for the Elmy challenge:
- Import the raw data with `Elmy_import`
- Import external data with `download_read_csv`
"""
from __future__ import annotations
from typing import Literal, Tuple, List
import numpy as np
import pandas as pd
import os
import urllib.request as url
from sklearn.preprocessing import RobustScaler
import time

import sys
sys.path.append('..')
from data_processing.args import opts
from interpolation.auto_regressive_interpolation import ar_interpolation_multiple_nan_seq, ar_interpolation

def Elmy_import(filepath: str, target=False, with_date=True):
    """
    Import data from a csv file and return a pandas dataframe
    """
    data = pd.read_csv(filepath, index_col=0)

    if not(target) and with_date:
        # Add a Date column (Year, Month, Day, Hour) with utc time
        data['Date (UTC)'] = pd.to_datetime(data.index, utc=True)
        # Add a Date column (Year, Month, Day) with Europe/Paris time
        data['Date'] = pd.to_datetime(data.index.to_series().apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')))
    return data
    
def test_equal(X_raw: pd.DataFrame, X: pd.DataFrame) -> None:
    print(f"Same index:   {X_raw.index.equals(X.index)}")
    X_loc = X.loc[X_raw.index, X_raw.columns]
    print(f"Same columns: {X_raw.columns.equals(X_loc.columns)}")
    print(f"Same values:  {X_raw.equals(X_loc)}")


def download_read_csv(
    filepath: Literal['../data/external/temperature-quotidienne-departementale.csv', 
                      '../data/external/pic-journalier-consommation-brute.csv', 
                      '../data/external/extremas-quotidiens-flux-commerciaux.csv', 
                      '../data/external/courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration.csv', 
                      '../data/external/consommation-quotidienne-brute.csv',
                      '../data/external/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv',
                      '../data/external/prod-nat-gaz-horaire-def.csv']
) -> pd.DataFrame:
    if not os.path.exists('../data/external'):
        os.makedirs('../data/external')   
    if not os.path.exists(filepath):
        if filepath=='../data/external/temperature-quotidienne-departementale.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/temperature-quotidienne-departementale/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B",
                filename=filepath
            )
        elif filepath=='../data/external/pic-journalier-consommation-brute.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/pic-journalier-consommation-brute/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B",
                filename=filepath
            )
        elif filepath=='../data/external/extremas-quotidiens-flux-commerciaux.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/extremas-quotidiens-flux-commerciaux/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
                filename=filepath
            )
        elif filepath=='../data/external/courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
                filename=filepath
            )
        elif filepath=='../data/external/consommation-quotidienne-brute.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/consommation-quotidienne-brute/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B",
                filename=filepath
            )
        elif filepath=='../data/external/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/rayonnement-solaire-vitesse-vent-tri-horaires-regionaux/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B",
                filename=filepath
            )
        elif filepath=='../data/external/prod-nat-gaz-horaire-def.csv':
            url.urlretrieve(
                url="https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/prod-nat-gaz-horaire-def/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B",
                filename=filepath
            )
    return pd.read_csv(filepath, sep=';')


def download_file_from_google_drive(gdrive_link, output_file):
    # Extract the FILEID from the Google Drive link
    fileid_index = gdrive_link.find('/d/') + 3
    fileid = gdrive_link[fileid_index: gdrive_link.find('/', fileid_index)]
    
    # Download the file using urllib.request.urlretrieve()
    try:
        url.urlretrieve(url=f"https://docs.google.com/uc?export=download&id={fileid}", filename=output_file)
        print(f"File downloaded successfully as {output_file}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def download_raw_data(
    filepath: Literal['../data/raw/X_train_raw.csv',
                      '../data/raw/y_train_raw.csv',
                      '../data/raw/X_test_raw.csv',
                      '../data/raw/y_test_random.csv'],
    with_date=True,
):
    if not os.path.exists('../data/raw'):
        os.makedirs('../data/raw')   
    if not os.path.exists(filepath):
        if filepath=='../data/raw/X_train_raw.csv':
            download_file_from_google_drive('https://drive.google.com/file/d/1EEIkrogO4g3PYsWBmx5ZkRO4Yo4-MGaT/view?usp=drive_link', filepath)
        elif filepath=='../data/raw/y_train_raw.csv':
            download_file_from_google_drive('https://drive.google.com/file/d/1IP4XOAixeI3NYpReGHpdn_Eue74O-SWf/view?usp=drive_link', filepath)
        if filepath=='../data/raw/X_test_raw.csv':
            download_file_from_google_drive('https://drive.google.com/file/d/1q4w0CrPKQTmtuFhloahfhxEgzPVrhiGE/view?usp=drive_link', filepath)
        elif filepath=='../data/raw/y_test_random.csv':
            download_file_from_google_drive('https://drive.google.com/file/d/1PMxELlBIM9GG3rTwv7-RpN3NycOonrmT/view?usp=drive_link', filepath)
    data = pd.read_csv(filepath, index_col=0)
    if with_date:
        # Add a Date column (Year, Month, Day, Hour) with utc time
        data['Date (UTC)'] = pd.to_datetime(data.index, utc=True)
        # Add a Date column (Year, Month, Day) with Europe/Paris time
        data['Date'] = pd.to_datetime(data.index.to_series().apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')))
    return data      


def add_external_feature(
    feature: Literal['temperature-quotidienne-departementale', 
                      'pic-journalier-consommation-brute', 
                      'extremas-quotidiens-flux-commerciaux', 
                      'courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration', 
                      'consommation-quotidienne-brute',
                      'rayonnement-solaire-vitesse-vent-tri-horaires-regionaux',
                      'prod-nat-gaz-horaire-def',
                      'Electricity_consumption_actual_and_forecast',
                      'Spot_prices_all_countries'],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_to_path = lambda x: f'../data/external/{x}.csv',
):
    if feature == 'temperature-quotidienne-departementale':
        temperature_quotidienne_departementale = download_read_csv(feature_to_path(feature))
        # Take the mean over departments
        temperature_quotidienne = temperature_quotidienne_departementale.groupby('Date')[['TMin (°C)', 'TMax (°C)', 'TMoy (°C)']].mean()
        temperature_quotidienne.index = pd.to_datetime(temperature_quotidienne.index, format='%Y-%m-%d')
        # Merge the daily average temperature with the training and test set
        X_train_new = X_train.merge(temperature_quotidienne, on='Date', how='left')
        X_test_new = X_test.merge(temperature_quotidienne, on='Date', how='left')
    elif feature == 'pic-journalier-consommation-brute':
        pic_journalier_consommation_brute = download_read_csv(feature_to_path(feature))
        pic_journalier_consommation_brute['Date'] = pd.to_datetime(pic_journalier_consommation_brute['Date'], format='%Y-%m-%d')
        X_train_new = X_train.merge(pic_journalier_consommation_brute, on='Date', how='left')
        X_test_new = X_test.merge(pic_journalier_consommation_brute, on='Date', how='left')
    elif feature == 'extremas-quotidiens-flux-commerciaux':
        extremas_quotidiens_flux_commerciaux = download_read_csv(feature_to_path(feature))
        extremas_quotidiens_flux_commerciaux.drop(columns="Temperature moy (°C)", inplace=True)
        extremas_quotidiens_flux_commerciaux['Date'] = pd.to_datetime(extremas_quotidiens_flux_commerciaux['Date'], format='%Y-%m-%d')
        X_train_new = X_train.merge(extremas_quotidiens_flux_commerciaux, on='Date', how='left')
        X_test_new = X_test.merge(extremas_quotidiens_flux_commerciaux, on='Date', how='left')
    elif feature == 'courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration':
        production_mensuelle_energie_eolienne = download_read_csv(feature_to_path(feature)).dropna()
        production_mensuelle_energie_eolienne['Date (UTC)'] = pd.to_datetime(pd.to_datetime(production_mensuelle_energie_eolienne['Date'] + '-' + production_mensuelle_energie_eolienne.Heure).apply(lambda x: x.tz_localize('Europe/Paris', ambiguous='NaT')), utc=True)
        production_mensuelle_energie_eolienne.drop(['Heure', 'Date'], axis=1, inplace=True)
        X_train_new = X_train.merge(production_mensuelle_energie_eolienne, on='Date (UTC)', how='left')
        X_test_new = X_test.merge(production_mensuelle_energie_eolienne, on='Date (UTC)', how='left')
    elif feature == 'consommation-quotidienne-brute':
        consumption = download_read_csv(feature_to_path(feature)).dropna()
        consumption['Date (UTC)'] = pd.to_datetime(consumption['Date - Heure'], utc=True)
        # Do not include 'Consommation brute totale (MW)' as it is the sum of 'Consommation brute gaz totale' and 'Consommation brute électricité'
        consumption = consumption[['Date (UTC)', 'Consommation brute gaz totale (MW PCS 0°C)', 'Consommation brute électricité (MW) - RTE']]
        consumption.drop_duplicates(subset='Date (UTC)', keep='first', inplace=True)
        X_train_new = X_train.merge(consumption, on='Date (UTC)', how='left')
        X_test_new = X_test.merge(consumption, on='Date (UTC)', how='left')
    elif feature == 'rayonnement-solaire-vitesse-vent-tri-horaires-regionaux':
        regional_solar_and_wind_power = download_read_csv(feature_to_path(feature))
        regional_solar_and_wind_power['Date (UTC)'] = pd.to_datetime(regional_solar_and_wind_power['Date'], utc=True)
        solar_and_wind_power = regional_solar_and_wind_power.groupby('Date (UTC)')[['Vitesse du vent à 100m (m/s)', 'Rayonnement solaire global (W/m2)']].mean()
        del regional_solar_and_wind_power
        X_train_new = X_train.merge(solar_and_wind_power, on='Date (UTC)', how='left')
        X_test_new = X_test.merge(solar_and_wind_power, on='Date (UTC)', how='left')
    elif feature == 'prod-nat-gaz-horaire-def':
        prod_nat_gaz_horaire = download_read_csv(feature_to_path(feature))
        prod_nat_gaz_horaire.index = prod_nat_gaz_horaire['Date']
        prod_nat_gaz_horaire = prod_nat_gaz_horaire.drop('Date', axis=1).loc[:, '00_00_00': '23_00_00']
        prod_nat_gaz_horaire.columns = prod_nat_gaz_horaire.columns.str.replace('_', ':')
        melted_df = prod_nat_gaz_horaire.reset_index().melt(id_vars='Date', var_name='Hour', value_name='Production horaire de biométhane (MWh - 0°C PCS)')
        del prod_nat_gaz_horaire
        melted_df['Date_and_hour'] = pd.to_datetime(melted_df['Date'].astype(str) + ' ' + melted_df['Hour'])
        melted_df['Date_and_hour'] = melted_df['Date_and_hour'].apply(localize_date)
        melted_df['Date (UTC)'] = pd.to_datetime(melted_df['Date_and_hour'], utc=True)
        melted_df = melted_df[['Date (UTC)', 'Production horaire de biométhane (MWh - 0°C PCS)']].dropna()
        melted_df = melted_df.groupby('Date (UTC)')[['Production horaire de biométhane (MWh - 0°C PCS)']].sum()
        X_train_new = X_train.merge(melted_df, on='Date (UTC)', how='left')
        X_test_new = X_test.merge(melted_df, on='Date (UTC)', how='left')
    elif feature == 'Electricity_consumption_actual_and_forecast':
        data_load_fr = pd.read_csv(feature_to_path('Total Load - Day Ahead _ Actual_Fr'), quoting = 3)
        data_load_fr.columns = ['Date', 'Load_Forecast_Fr', 'Load_Actual_Fr']
        data_load_fr['Date'] = pd.to_datetime(data_load_fr['Date'].str[1:17], format='%d.%m.%Y %H:%M')
        data_load_fr['Load_Forecast_Fr'] = data_load_fr['Load_Forecast_Fr'].str[2:-2]
        data_load_fr['Load_Forecast_Fr'] = pd.to_numeric(data_load_fr['Load_Forecast_Fr'], errors='coerce')
        data_load_fr['Load_Actual_Fr'] = data_load_fr['Load_Actual_Fr'].str[2:-3]
        data_load_fr['Load_Actual_Fr'] = pd.to_numeric(data_load_fr['Load_Actual_Fr'], errors='coerce')
        data_load_ge = pd.read_csv(feature_to_path('Total Load - Day Ahead _ Actual_Ge'), quoting = 3)
        data_load_ge.columns = ['Date', 'Load_Forecast_Ge', 'Load_Actual_Ge']
        data_load_ge['Load_Forecast_Ge'] = data_load_ge['Load_Forecast_Ge'].str[2:-2]
        data_load_ge['Load_Forecast_Ge'] = pd.to_numeric(data_load_ge['Load_Forecast_Ge'], errors='coerce')
        data_load_ge['Load_Actual_Ge'] = data_load_ge['Load_Actual_Ge'].str[2:-3]
        data_load_ge['Load_Actual_Ge'] = pd.to_numeric(data_load_ge['Load_Actual_Ge'], errors='coerce')
        data_load_ge['Date'] = pd.to_datetime(data_load_ge['Date'].str[1:17], format='%d.%m.%Y %H:%M')
        data_load_ge.set_index('Date', inplace=True)
        data_load_ge = data_load_ge.resample('H').mean()
        data_load_ge.reset_index(inplace=True)
        data_load_ge_fr = data_load_fr.merge(data_load_ge, on='Date', how='left')
        data_load_ge_fr.dropna(inplace=True)
        data_load_ge_fr['Date'] = data_load_ge_fr['Date'].dt.tz_localize('Europe/Paris', ambiguous='NaT')
        data_load_ge_fr['Date (UTC)'] = data_load_ge_fr['Date'].dt.tz_convert('UTC')
        data_load_ge_fr.drop('Date', axis=1, inplace=True)
        # Load_Actual_Fr and Consommation brute électricité (MW) - RTE are the same -> create a new feature based on the actual - forecast
        data_load_ge_fr['delta_Load_Fr'] = data_load_ge_fr['Load_Actual_Fr'] - data_load_ge_fr['Load_Forecast_Fr']
        data_load_ge_fr.drop(['Load_Actual_Fr', 'Load_Forecast_Fr'], axis=1, inplace=True)
        data_load_ge_fr['delta_Load_Ge'] = data_load_ge_fr['Load_Actual_Ge'] - data_load_ge_fr['Load_Forecast_Ge']
        data_load_ge_fr.drop('Load_Forecast_Ge', axis=1, inplace=True) # keep the actual in Germany
        X_train_new = X_train.merge(data_load_ge_fr, on='Date (UTC)', how='left')
        X_test_new = X_test.merge(data_load_ge_fr, on='Date (UTC)', how='left') 
    elif feature == 'Spot_prices_all_countries':
        try:
            spot_price = pd.read_csv(feature_to_path(feature))
            spot_price['Date (UTC)'] = pd.to_datetime(spot_price['Datetime (UTC)'], format='%Y-%m-%d %H:%M:%S', utc=True)
            print(f"Loaded Noe's version of the Spot_prices_all_countries")
        except:
            spot_price = pd.read_csv(feature_to_path(feature), header=None, names=['Country', 'ISO3 Code', 'Datetime (UTC)', 'Datetime (Local)', 'Price (EUR/MWhe)'])
            spot_price[['Country', 'ISO3 Code', 'Datetime (UTC)', 'Datetime (Local)', 'Price (EUR/MWhe)']] = spot_price['Country'].str.split(',', expand=True)
            spot_price = spot_price.iloc[1:]
            spot_price['Date (UTC)'] = pd.to_datetime(spot_price['Datetime (UTC)'], format='%Y-%m-%d %H:%M:%S', utc=True)
            print(f"Loaded Tim's version of the Spot_prices_all_countries")
        pivot_spot_price = spot_price.pivot(index= 'Date (UTC)', columns='Country', values='Price (EUR/MWhe)').astype(float)
        # drop Hungary as it contains many NaN
        pivot_spot_price.drop('Hungary', axis=1, inplace=True)
        X_train_new = X_train.merge(pivot_spot_price, on='Date (UTC)', how='left')
        X_test_new = X_test.merge(pivot_spot_price, on='Date (UTC)', how='left')

    X_train_new.index = X_train.index
    X_test_new.index = X_test.index
    return X_train_new, X_test_new

def localize_date(x: pd.Timestamp):
    try:
        return x.tz_localize('Europe/Paris', ambiguous='NaT')
    except:
        return pd.NaT  # Return NaT if NonExistentTimeError occurs
    
def clean_train_data(X_train: pd.DataFrame, columns_to_clean):
    X_train_clean = X_train.drop(columns_to_clean, axis=1).copy()

    # TMoy (°C)
    X_train_clean['TMoy (°C)'] = (X_train['TMoy (°C)'] + X_train['Température moyenne (°C)']) / 2.0

    # wind_power_forecasts_average
    wind_power_forecasts_average_clean = X_train['wind_power_forecasts_average'].copy()
    wind_power_forecasts_average_clean[X_train['wind_power_forecasts_average'].isna()] = X_train['prod_eolienne_MWh'][X_train['wind_power_forecasts_average'].isna()]
    X_train_clean['wind_power_forecasts_average'] = wind_power_forecasts_average_clean

    # solar_power_forecasts_average
    replacing_day = '2023-02-21'
    missing_day = '2023-02-20'
    scaling_ratio = X_train[X_train['Date']==missing_day]['prod_solaire_MWh'].max() / X_train[X_train['Date']==replacing_day]['prod_solaire_MWh'].max()
    replacing_values = (scaling_ratio * X_train[X_train['Date']==replacing_day]['solar_power_forecasts_average']).values
    solar_power_forecasts_average_clean = X_train['solar_power_forecasts_average'].copy()
    solar_power_forecasts_average_clean[X_train['solar_power_forecasts_average'].isna()] = replacing_values
    X_train_clean['solar_power_forecasts_average'] = solar_power_forecasts_average_clean

    # Consommation brute électricité (MW) - RTE
    consommation_brute_elec_clean = X_train['Consommation brute électricité (MW) - RTE'].copy()
    consommation_brute_elec_clean[X_train['Consommation brute électricité (MW) - RTE'].isna()] = X_train[X_train['Consommation brute électricité (MW) - RTE'].isna()]['load_forecast']
    X_train_clean['Consommation brute électricité (MW) - RTE'] = consommation_brute_elec_clean

    # wind_power_forecasts_std
    neighbour_dates = ['2022-12-20', '2022-12-21', '2022-12-22', '2022-12-24', '2022-12-25', '2022-12-26']
    missing_day = '2022-12-23'
    wind_power_forecasts_std_clean = X_train['wind_power_forecasts_std'].copy()
    df = X_train[X_train['Date'].isin(neighbour_dates)][['wind_power_forecasts_std']]
    df['Hour'] = pd.to_datetime(df.index).hour
    wind_power_forecasts_std_clean[X_train['Date']==missing_day] = df.groupby('Hour')['wind_power_forecasts_std'].mean().values
    X_train_clean['wind_power_forecasts_std'] = wind_power_forecasts_std_clean

    # solar_power_forecasts_std
    neighbour_dates = ['2023-02-17', '2023-02-18', '2023-02-19', '2023-02-21', '2023-02-22', '2023-02-23']
    missing_day = '2023-02-20'
    solar_power_forecasts_std_clean = X_train['solar_power_forecasts_std'].copy()
    df = X_train[X_train['Date'].isin(neighbour_dates)][['solar_power_forecasts_std']]
    df['Hour'] = pd.to_datetime(df.index).hour
    solar_power_forecasts_std_clean[X_train['Date']==missing_day] = df.groupby('Hour')['solar_power_forecasts_std'].mean().values
    X_train_clean['solar_power_forecasts_std'] = solar_power_forecasts_std_clean

    # coal_power_available
    X_train_clean['coal_power_available'] = X_train['coal_power_available'].ffill()

    # gas_power_available
    X_train_clean['gas_power_available'] = X_train['gas_power_available'].ffill()

    # nucelear_power_available
    X_train_clean['nucelear_power_available'] = X_train['nucelear_power_available'].ffill()

    # Production horaire de biométhane (MWh - 0°C PCS)
    X_train_clean['Production horaire de biométhane (MWh - 0°C PCS)'] = X_train['Production horaire de biométhane (MWh - 0°C PCS)'].ffill()

    # Consommation brute gaz totale (MW PCS 0°C)
    missing_days = ['2022-12-04', '2022-12-05']
    neighbour_dates = ['2022-12-01', '2022-12-02', '2022-12-03', '2022-12-06', '2022-12-07', '2022-12-08']
    conso_brute_gaz_clean = X_train['Consommation brute gaz totale (MW PCS 0°C)'].interpolate(method='linear')
    df = X_train[X_train['Date'].isin(neighbour_dates)][['Consommation brute gaz totale (MW PCS 0°C)']]
    df['Hour'] = pd.to_datetime(df.index).hour
    for missing_day in missing_days:
        conso_brute_gaz_clean[X_train['Date']==missing_day] = df.groupby('Hour')['Consommation brute gaz totale (MW PCS 0°C)'].mean().values
    X_train_clean['Consommation brute gaz totale (MW PCS 0°C)'] = conso_brute_gaz_clean

    # Vitesse du vent à 100m (m/s)
    vitesse_du_vent_clean = X_train['Vitesse du vent à 100m (m/s)'].copy()
    vitesse_du_vent_clean.index = range(len(vitesse_du_vent_clean))
    vitesse_du_vent_clean = vitesse_du_vent_clean.interpolate(method='polynomial', order=3).bfill()
    vitesse_du_vent_clean.index = X_train.index
    X_train_clean['Vitesse du vent à 100m (m/s)'] = vitesse_du_vent_clean

    # Rayonnement solaire global (W/m2)
    rayonnement_solaire_clean = X_train['Rayonnement solaire global (W/m2)'].copy()
    rayonnement_solaire_clean.index = range(len(rayonnement_solaire_clean))
    rayonnement_solaire_clean = (rayonnement_solaire_clean.interpolate(method='polynomial', order=1).bfill()).abs()
    rayonnement_solaire_clean.index = X_train.index
    X_train_clean['Rayonnement solaire global (W/m2)'] = rayonnement_solaire_clean

    return X_train_clean

def clean_test_data(X_test: pd.DataFrame, columns_to_clean):
    X_test_clean = X_test.drop(columns_to_clean, axis=1).copy()

    # TMoy (°C)
    X_test_clean['TMoy (°C)'] = (X_test['TMoy (°C)'] + X_test['Température moyenne (°C)']) / 2.0

    # wind_power_forecasts_average
    X_test_clean['wind_power_forecasts_average'] = X_test['wind_power_forecasts_average'].copy()

    # solar_power_forecasts_average
    replacing_day = '2023-04-26'
    missing_day = '2023-04-27'
    scaling_ratio = X_test[X_test['Date']==missing_day]['prod_solaire_MWh'].max() / X_test[X_test['Date']==replacing_day]['prod_solaire_MWh'].max()
    replacing_values = (scaling_ratio * X_test[X_test['Date']==replacing_day]['solar_power_forecasts_average']).values
    solar_power_forecasts_average_clean = X_test['solar_power_forecasts_average'].copy()
    solar_power_forecasts_average_clean[X_test['solar_power_forecasts_average'].isna()] = replacing_values
    X_test_clean['solar_power_forecasts_average'] = solar_power_forecasts_average_clean

    # Consommation brute électricité (MW) - RTE
    X_test_clean['Consommation brute électricité (MW) - RTE'] = X_test['Consommation brute électricité (MW) - RTE'].copy()

    # wind_power_forecasts_std
    X_test_clean['wind_power_forecasts_std'] = X_test['wind_power_forecasts_std'].copy()

    # solar_power_forecasts_std
    neighbour_dates = ['2023-04-25', '2023-04-26', '2023-04-28', '2023-04-29']
    missing_day = '2023-04-27'
    solar_power_forecasts_std_clean = X_test['solar_power_forecasts_std'].copy()
    df = X_test[X_test['Date'].isin(neighbour_dates)][['solar_power_forecasts_std']]
    df['Hour'] = pd.to_datetime(df.index).hour
    solar_power_forecasts_std_clean[X_test['Date']==missing_day] = df.groupby('Hour')['solar_power_forecasts_std'].mean().values
    X_test_clean['solar_power_forecasts_std'] = solar_power_forecasts_std_clean

    # coal_power_available
    X_test_clean['coal_power_available'] = X_test['coal_power_available'].copy()

    # gas_power_available
    X_test_clean['gas_power_available'] = X_test['gas_power_available'].copy()

    # nucelear_power_available
    X_test_clean['nucelear_power_available'] = X_test['nucelear_power_available'].copy()

    # Production horaire de biométhane (MWh - 0°C PCS)
    X_test_clean['Production horaire de biométhane (MWh - 0°C PCS)'] = X_test['Production horaire de biométhane (MWh - 0°C PCS)'].copy()

    # Consommation brute gaz totale (MW PCS 0°C)
    X_test_clean['Consommation brute gaz totale (MW PCS 0°C)'] = X_test['Consommation brute gaz totale (MW PCS 0°C)'].copy()

    # Vitesse du vent à 100m (m/s)
    vitesse_du_vent_clean = X_test['Vitesse du vent à 100m (m/s)'].copy()
    vitesse_du_vent_clean.index = range(len(vitesse_du_vent_clean))
    vitesse_du_vent_clean = vitesse_du_vent_clean.interpolate(method='polynomial', order=3).bfill()
    vitesse_du_vent_clean.index = X_test.index
    X_test_clean['Vitesse du vent à 100m (m/s)'] = vitesse_du_vent_clean

    # Rayonnement solaire global (W/m2)
    rayonnement_solaire_clean = X_test['Rayonnement solaire global (W/m2)'].copy()
    rayonnement_solaire_clean.index = range(len(rayonnement_solaire_clean))
    rayonnement_solaire_clean = (rayonnement_solaire_clean.interpolate(method='polynomial', order=1).bfill()).abs()
    rayonnement_solaire_clean.index = X_test.index
    X_test_clean['Rayonnement solaire global (W/m2)'] = rayonnement_solaire_clean

    return X_test_clean


def interpolate_psp_train(predicted_spot_price: pd.Series, start_date='2023-01-03 01:00:00+01:00'):
    new_psp = predicted_spot_price.copy()
    psp_with_nan = predicted_spot_price.loc[start_date:].copy()
    del predicted_spot_price
    psp_with_nan.index = np.arange(len(psp_with_nan))
    autoreg_interpolation, _ = ar_interpolation_multiple_nan_seq(psp_with_nan, p=70, polynomial_order=1, with_tqdm=True)
    new_psp.loc[start_date:] = autoreg_interpolation.values
    return new_psp

def interpolate_psp_test(predicted_spot_price: pd.Series, start_idx=51):
    new_psp = predicted_spot_price.copy()
    psp_with_nan = predicted_spot_price.iloc[start_idx:].copy()
    del predicted_spot_price
    psp_with_nan.index = np.arange(len(psp_with_nan))
    autoreg_interpolation, _ = ar_interpolation_multiple_nan_seq(psp_with_nan, p=70, polynomial_order=1, with_tqdm=True)
    new_psp.iloc[start_idx:] = autoreg_interpolation.values
    return new_psp


def interpolate_psp_start_test(train_psp: pd.Series, test_psp: pd.Series):
    new_test_psp = test_psp.copy()
    full_psp = pd.concat((train_psp[train_psp.notna()], test_psp)).copy()
    index_full_psp = full_psp.index.copy()
    del train_psp, test_psp
    full_psp.index = np.arange(len(full_psp))
    ar_interp, _ = ar_interpolation(full_psp, p=70, polynomial_order=1)
    ar_interp_psp = pd.Series(ar_interp.values, index=index_full_psp)
    new_test_psp.loc[new_test_psp.index[0]:index_full_psp[-1]] = ar_interp_psp.loc[new_test_psp.index[0]:index_full_psp[-1]].values
    return new_test_psp

def add_cyclic_feature(df: pd.DataFrame, col_name: str, period: int):
    df[f'x_{col_name}'] = np.cos(2 * np.pi * df[col_name] / period)
    df[f'y_{col_name}'] = np.sin(2 * np.pi * df[col_name] / period)
    return df

def add_date_features(df: pd.DataFrame, date_column='Date (UTC)'):
    df[date_column] = pd.to_datetime(df[date_column])
    df['day_of_month'] = df[date_column].dt.day
    df = add_cyclic_feature(df, 'day_of_month', 31)
    df['day_of_week'] = df[date_column].dt.dayofweek
    df = add_cyclic_feature(df, 'day_of_week', 7)
    df['month'] = df[date_column].dt.month
    df = add_cyclic_feature(df, 'month', 12)
    df['quarter'] = df[date_column].dt.quarter
    df = add_cyclic_feature(df, 'quarter', 4)
    #df['year'] = df[date_column].dt.year
    df['hour_of_day'] = df[date_column].dt.hour
    df = add_cyclic_feature(df, 'hour_of_day', 24)
    df['week_number'] = df[date_column].dt.isocalendar().week
    df = add_cyclic_feature(df, 'week_number', 52)
    df['time_of_day'] = pd.cut(df['hour_of_day'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
    df = pd.get_dummies(df, columns=['time_of_day'], prefix='time_of_day', drop_first=True, dtype=float)  # One-hot encode 'time_of_day'
    df['season'] = (df['month'] % 12 + 3) // 3 # 1:spring, 2:summer, 3:fall, 4:winter
    df = add_cyclic_feature(df, 'season', 4)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df.drop(['day_of_month', 'day_of_week', 'month', 'quarter', 'hour_of_day', 'week_number', 'season'], axis=1, inplace=True)
    return df

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: RobustScaler):
    X_train['predicted_spot_price'].fillna(0, inplace=True)
    X_train.drop(['Date', 'Date (UTC)'], axis=1, inplace=True)
    X_test.drop(['Date', 'Date (UTC)'], axis=1, inplace=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    return X_train_scaled, X_test_scaled


def main(external_features: List[str], interpolate_psp: bool):
    X_train_raw = download_raw_data('../data/raw/X_train_raw.csv', with_date=True)
    y_train_raw = download_raw_data('../data/raw/y_train_raw.csv', with_date=False)
    X_test_raw = download_raw_data('../data/raw/X_test_raw.csv', with_date=True)
    y_test_random = download_raw_data('../data/raw/y_test_random.csv', with_date=False)
    print(f'X_train_raw shape: {X_train_raw.shape} - y_train_raw shape: {y_train_raw.shape}')
    print(f'X_test_raw shape: {X_test_raw.shape} - y_test_raw shape: {y_test_random.shape}')

    X_train = X_train_raw.copy(); del X_train_raw
    X_test = X_test_raw.copy(); del X_test_raw
    for external_feature in external_features:
        X_train, X_test = add_external_feature(external_feature, X_train, X_test)
        print(f"Successfully added {external_feature}!")

    columns_to_clean = ['Température moyenne (°C)', 'TMoy (°C)', 'prod_eolienne_MWh', 'wind_power_forecasts_average', 'prod_solaire_MWh', 'solar_power_forecasts_average', 'Consommation brute électricité (MW) - RTE', 'load_forecast', 'wind_power_forecasts_std', 'solar_power_forecasts_std', 'coal_power_available', 'gas_power_available', 'nucelear_power_available', 'Production horaire de biométhane (MWh - 0°C PCS)', 'Consommation brute gaz totale (MW PCS 0°C)', 'Vitesse du vent à 100m (m/s)', 'Rayonnement solaire global (W/m2)']
    X_train = clean_train_data(X_train, columns_to_clean)
    X_test = clean_test_data(X_test, columns_to_clean)

    if interpolate_psp:
        X_train['predicted_spot_price'] = interpolate_psp_train(X_train['predicted_spot_price'])
        X_test['predicted_spot_price'] = interpolate_psp_test(X_test['predicted_spot_price'])
        X_test['predicted_spot_price'] = interpolate_psp_start_test(X_train['predicted_spot_price'], X_test['predicted_spot_price'])
        print(f"Successfully interpolated predicted_spot_price!")

    X_train = add_date_features(X_train)
    X_test = add_date_features(X_test)
    print(f"Successfully added date features!")

    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')
    X_train.to_csv('../data/processed/X_train.csv')
    X_test.to_csv('../data/processed/X_test.csv')

    scaler = RobustScaler()
    X_train, X_test = scale_data(X_train, X_test, scaler)

    X_train.to_csv('../data/processed/X_train_scaled.csv')
    X_test.to_csv('../data/processed/X_test_scaled.csv')
    print("Successfully saved the data at ../data/processed/ (scaled and unscaled)!")


if __name__=='__main__':
    args = opts()
    start_time = time.time()
    main(args.external_features, args.interpolate_psp)
    print(f"Total import time: {time.time() - start_time}s")

