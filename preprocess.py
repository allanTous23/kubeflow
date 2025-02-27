import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame en supprimant certaines colonnes et en imputant les valeurs manquantes.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données brutes.
    
    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    df = df.copy()
    
    # Suppression des colonnes avec trop de valeurs manquantes
    columns_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Imputation des catégories manquantes
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('No Fireplace')
    
    # Imputation de LotFrontage par la médiane
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # Imputation des colonnes Garage par "No Garage"
    garage_columns = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']
    df[garage_columns] = df[garage_columns].fillna('No Garage')
    
    # Imputation des colonnes Bsmt par "No Basement"
    bsmt_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    df[bsmt_columns] = df[bsmt_columns].fillna('No Basement')
    
    # Trouver la valeur la plus fréquente (modale)
    most_frequent_value = df['MSZoning'].mode()[0]  # [0] pour récupérer la valeur
    most_frequent_value_SaleType = df['SaleType'].mode()[0]

    # Remplacer les NaN par la valeur la plus fréquente
    df['MSZoning'] = df['MSZoning'].fillna(most_frequent_value)
    df['SaleType'] = df['SaleType'].fillna(most_frequent_value_SaleType)
    
    # Imputation de MasVnrArea par 0
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    return df
