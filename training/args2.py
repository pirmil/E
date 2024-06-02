from argparse import ArgumentParser, Namespace

"""
col_d1_of_sd24 = ['Température référence (°C)']
col_sd24 = ['TMin (°C)', 'TMax (°C)', 'TMoy (°C)', 'Pic journalier consommation (MW)', 'Rayonnement solaire global (W/m2)', 'Solde min (MW)', 'Solde max (MW)', 'solar_power_forecasts_average']
#col_d1 = ['coal_power_available', 'gas_power_available', 'nucelear_power_available', 'Production horaire de biométhane (MWh - 0°C PCS)', 'predicted_spot_price', 'Consommation brute gaz totale (MW PCS 0°C)', 'Consommation brute électricité (MW) - RTE', 'solar_power_forecasts_average', 'wind_power_forecasts_average', 'solar_power_forecasts_std', 'wind_power_forecasts_std']
col_d1 = ['coal_power_available', 'gas_power_available', 'nucelear_power_available', 'Production horaire de biométhane (MWh - 0°C PCS)', 'predicted_spot_price', 'Consommation brute gaz totale (MW PCS 0°C)', 'Consommation brute électricité (MW) - RTE', 'solar_power_forecasts_average', 'wind_power_forecasts_average', 'solar_power_forecasts_std', 'wind_power_forecasts_std']
col_d2 = ['Vitesse du vent à 100m (m/s)']
to_drop = []
col_drop = list(set(col_d1_of_sd24).union(col_sd24).union(col_d1).union(col_d2).union(to_drop))
"""
col_d1 = col_d2 = col_sd24 = col_d1_of_sd24 = []
col_drop = []
# 39 features
col_keep = ['TMin (°C)', 'TMax (°C)', 'Pic journalier consommation (MW)',
       'Température référence (°C)', 'Solde min (MW)', 'Solde max (MW)',
       'Load_Actual_Ge', 'delta_Load_Fr', 'delta_Load_Ge', 'Austria',
       'Belgium', 'Bulgaria', 'Croatia', 'Czechia', 'Denmark', 'Estonia',
       'Finland', 'France', 'Germany', 'Greece', 'Spain', 'Switzerland',
       'TMoy (°C)', 'wind_power_forecasts_average',
       'solar_power_forecasts_average',
       'Consommation brute électricité (MW) - RTE',
       'wind_power_forecasts_std', 'solar_power_forecasts_std',
       'coal_power_available', 'gas_power_available',
       'nucelear_power_available',
       'Consommation brute gaz totale (MW PCS 0°C)',
       'Vitesse du vent à 100m (m/s)',
       'Rayonnement solaire global (W/m2)', 'x_day_of_week',
       'y_day_of_week', 'x_hour_of_day', 'y_hour_of_day', 'is_weekend']

"""
# 30 features
col_keep = ['TMin (°C)', 'TMax (°C)', 'Pic journalier consommation (MW)',
       'Température référence (°C)', 'Solde min (MW)', 'Solde max (MW)',
       'Load_Actual_Ge', 'delta_Load_Fr', 'delta_Load_Ge', 'France', 'Germany', 'Spain', 'Switzerland',
       'TMoy (°C)', 'wind_power_forecasts_average',
       'solar_power_forecasts_average',
       'Consommation brute électricité (MW) - RTE',
       'wind_power_forecasts_std', 'solar_power_forecasts_std',
       'coal_power_available', 'gas_power_available',
       'nucelear_power_available',
       'Consommation brute gaz totale (MW PCS 0°C)',
       'Vitesse du vent à 100m (m/s)',
       'Rayonnement solaire global (W/m2)', 'x_day_of_week',
       'y_day_of_week', 'x_hour_of_day', 'y_hour_of_day', 'is_weekend']
"""

c_percentiles = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def opts() -> Namespace:
    parser = ArgumentParser(description="Perform training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.02, help="Dropout")
    parser.add_argument("--train-size", type=float, default=0.8, help="Training/validation ratio")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--hidden-sizes", type=int, nargs='+', default=[15, 10], help="Dimensions of the hidden LSTM layers")
    parser.add_argument("--metrics", type=str, nargs='+', default=['accuracy', 'weighted_accuracy'], help="Metrics to track the evolution of training")
    parser.add_argument("--d1", type=str, nargs='+', default=col_d1, help="Perform first order difference")
    parser.add_argument("--d2", type=str, nargs='+', default=col_d2, help="Perform second order difference")
    parser.add_argument("--sd24", type=str, nargs='+', default=col_sd24, help="Perform seasonal difference with 24 lags")
    parser.add_argument("--d1-sd24", type=str, nargs='+', default=col_d1_of_sd24, help="Perform difference of seasonal difference with 24 lags")
    parser.add_argument("--drop", type=str, nargs='+', default=col_drop, help="Columns to drop")
    parser.add_argument("--keep", type=str, nargs='+', default=col_keep, help="Columns to keep. It has the priority over drop.")
    parser.add_argument("--train-val-weights-None", action='store_true', help="If True, then weighted accuracy should be equal to accuracy")
    parser.add_argument("--visualize-features", action='store_true', help="Whether to visualize the features")
    parser.add_argument("--sequence-length", type=int, default=48, help="Number of hours to look back when training LSTMs")
    parser.add_argument("--model", type=str, default="LSTM", help="Model to use")

    parser.add_argument("--save-true", action='store_true', help="Whether to save the true prediction as well")
    parser.add_argument("--task", type=str, default="regression", help="Whether to perform classification or regression")
    parser.add_argument("--optimizer", type=str, default="adam", help="Name of the optimizer")
    parser.add_argument("--fill-method", type=str, default="interpolation", help="Scaler for the features")
    parser.add_argument("--features-scaler", type=str, default="RobustScaler", help="Scaler for the features")
    parser.add_argument("--target-scaler", type=str, default="RobustScaler", help="Scaler for the target")
    parser.add_argument("--clip-percentile", type=float, default=4.0, help="Clip the values of the target (top and bottom)")
    parser.add_argument("--pc1", type=float, default=70.0, help="Force the percentile of class 1 in the final submission")
    parser.add_argument("--c-percentiles", type=str, nargs='+', default=c_percentiles, help="The list of percentiles to try (won't affect the final submission)")
    parser.add_argument('--num-workers', type=int, default=16, help="Number of workers for data loading")
    parser.add_argument("--random-val-set", action='store_true', help="Choose the validation set at random. Else, choose the last days of training for validation")
    args = parser.parse_args()
    return args