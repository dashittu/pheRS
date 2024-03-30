import pandas as pd

def get_phecode_occurrences(icd_occurrences, icd_phecode_map, dx_icd=None):
    """
    Maps ICD occurrences to phecodes by merging ICD occurrence data with a mapping table.

    Parameters:
    - icd_occurrences: pandas.DataFrame containing ICD code occurrences. Must have columns 'icd', 'flag', and 'person_id'.
    - icd_phecode_map: pandas.DataFrame containing the mapping from ICD codes to phecodes. Must have columns 'icd', 'flag', and 'phecode'.
    - dx_icd: Optional. pandas.DataFrame containing diagnostic ICD codes to be removed. Must have columns 'icd' and 'flag'. If None, no ICD codes are removed.

    Returns:
    - A pandas.DataFrame with columns 'person_id' and 'phecode', representing the occurrence of phecodes per person.
    """

    # Validation checks (simplified for this example)
    required_columns = ['icd', 'flag', 'person_id']
    if not set(required_columns).issubset(icd_occurrences.columns):
        raise ValueError("icd_occurrences must contain columns: 'icd', 'flag', 'person_id'")
    if not set(['icd', 'flag', 'phecode']).issubset(icd_phecode_map.columns):
        raise ValueError("icd_phecode_map must contain columns: 'icd', 'flag', 'phecode'")

    # Remove diagnostic codes if dx_icd is provided
    if dx_icd is not None and not dx_icd.empty:
        icd_occurrences = pd.merge(icd_occurrences, dx_icd, on=['icd', 'flag'], how='left', indicator=True)
        icd_occurrences = icd_occurrences[icd_occurrences['_merge'] == 'left_only'].drop(columns=['_merge'])

    # Merge icd occurrences with icd phecode map
    phecode_occurrences = pd.merge(icd_occurrences, icd_phecode_map[['icd', 'flag', 'phecode']], on=['icd', 'flag'])

    # Drop duplicates and unnecessary columns
    phecode_occurrences = phecode_occurrences[['person_id', 'phecode']].drop_duplicates()

    return phecode_occurrences


def get_scores(weights, disease_phecode_map):
    """
    Aggregates phecode-specific weights into overall disease risk scores for individuals.

    Parameters:
    - weights: pandas.DataFrame containing phecode weights for each person. Must have columns 'person_id', 'phecode', and 'w'.
    - disease_phecode_map: pandas.DataFrame containing mapping from phecodes to diseases. Must have columns 'phecode' and 'disease_id'.

    Returns:
    - pandas.DataFrame with columns 'person_id', 'disease_id', and 'score', representing the aggregated risk score of diseases per person.
    """

    # Validation checks (simplified for this example)
    if not set(['person_id', 'phecode', 'w']).issubset(weights.columns):
        raise ValueError("weights must contain columns: 'person_id', 'phecode', 'w'")
    if not set(['phecode', 'disease_id']).issubset(disease_phecode_map.columns):
        raise ValueError("disease_phecode_map must contain columns: 'phecode', 'disease_id'")

    # Merge weights with disease phecode map
    merged_data = pd.merge(weights, disease_phecode_map, on='phecode', how='inner')

    # Calculate scores by summing weights per person per disease
    scores = merged_data.groupby(['person_id', 'disease_id'])['w'].sum().reset_index()
    scores.rename(columns={'w': 'score'}, inplace=True)

    return scores


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def get_residual_scores(demos, scores, lm_formula):
    """
    Calculates residual scores for diseases by adjusting for demographics using a linear model.

    Parameters:
    - demos: pandas.DataFrame containing demographic data for each person. Must have a column 'person_id'.
    - scores: pandas.DataFrame containing disease scores for each person. Must have columns 'person_id', 'disease_id', and 'score'.
    - lm_formula: string representing the formula for the linear model, e.g., 'score ~ age + sex'.

    Returns:
    - pandas.DataFrame with columns 'person_id', 'disease_id', 'score', and 'resid_score', where 'resid_score' is the standardized residual from the linear model.
    """

    # Validation checks (simplified for this example)
    if 'person_id' not in demos.columns:
        raise ValueError("demos must contain column: 'person_id'")
    if not set(['person_id', 'disease_id', 'score']).issubset(scores.columns):
        raise ValueError("scores must contain columns: 'person_id', 'disease_id', 'score'")

    # Merge scores with demographic data
    r_input = pd.merge(scores, demos, on='person_id', how='inner')

    # Fit linear model and calculate residuals
    lm_model = smf.ols(formula=lm_formula, data=r_input).fit()
    r_input['resid_score'] = lm_model.get_influence().resid_studentized_internal

    # Rearrange and return the results
    r_scores = r_input[['person_id', 'disease_id', 'score', 'resid_score']]

    return r_scores
