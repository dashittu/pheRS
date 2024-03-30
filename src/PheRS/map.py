import pandas as pd

def map_disease_to_phecode(disease_hpo_map, hpo_phecode_map):
    """
    Maps diseases to phecodes based on their association with HPO terms.

    Parameters:
    - disease_hpo_map: pandas.DataFrame containing disease to HPO term mappings.
      Must have columns 'disease_id' and 'hpo_term_id'.
    - hpo_phecode_map: pandas.DataFrame containing HPO term to phecode mappings.
      Must have columns 'hpo_term_id' and 'phecode'.

    Returns:
    - A pandas.DataFrame with unique mappings from 'disease_id' to 'phecode'.
    """

    # Validate input dataframes
    for df, required_cols in zip([disease_hpo_map, hpo_phecode_map],
                                 [['disease_id', 'hpo_term_id'], ['hpo_term_id', 'phecode']]):
        if not set(required_cols).issubset(df.columns):
            raise ValueError(f"DataFrame is missing one of the required columns: {required_cols}")
        if df[required_cols].isnull().any().any():
            raise ValueError("Input DataFrame contains missing values in the required columns.")

    # Merge the two dataframes on 'hpo_term_id'
    disease_phecode_map = pd.merge(disease_hpo_map, hpo_phecode_map, on='hpo_term_id')

    # Drop duplicates to ensure uniqueness and select only the required columns
    disease_phecode_map = disease_phecode_map[['disease_id', 'phecode']].drop_duplicates()

    # Optionally, you could sort the DataFrame for easier readability
    disease_phecode_map.sort_values(by=['disease_id', 'phecode'], inplace=True)

    return disease_phecode_map
