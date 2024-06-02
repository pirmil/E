from argparse import ArgumentParser, Namespace

default_features = ['temperature-quotidienne-departementale', 
                      'pic-journalier-consommation-brute', 
                      'extremas-quotidiens-flux-commerciaux', 
                      'courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration', 
                      'consommation-quotidienne-brute',
                      'rayonnement-solaire-vitesse-vent-tri-horaires-regionaux',
                      'prod-nat-gaz-horaire-def',
                      'Electricity_consumption_actual_and_forecast',
                      'Spot_prices_all_countries']

def opts() -> Namespace:
    parser = ArgumentParser(description="Import data and process it")
    parser.add_argument("--external-features", type=str, nargs='+', default=default_features, help="External features to add to the dataset")
    parser.add_argument("--interpolate-psp", action='store_true', help="Whether to interpolate the predicted spot price")
    args = parser.parse_args()
    return args