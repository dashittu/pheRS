import pandas as pd
import numpy as np


def get_dx_status(demos, icd_occurrences, min_unique_ages=2, disease_dx_icd_map=None):
    """
    Calculate dx_status for each person and disease based on ICD occurrences.

    :param demos: DataFrame containing demographic information with 'person_id'.
    :param icd_occurrences: DataFrame containing ICD code occurrences with 'person_id', 'icd', 'flag', 'occurrence_age'.
    :param min_unique_ages: Minimum number of unique ages at which a diagnosis must be recorded to set dx_status to 1.
    :param disease_dx_icd_map: DataFrame mapping disease IDs to ICD codes with 'disease_id', 'icd', 'flag'.
    :return: DataFrame with 'person_id', 'disease_id', 'dx_status'.
    """
    # Validate inputs (Placeholder for actual checks)
    # check_demos(demos)
    # check_icd_occurrences(icd_occurrences)
    # Check for numeric 'occurrence_age' in icd_occurrences and positive 'min_unique_ages'

    if disease_dx_icd_map is None:
        raise ValueError("disease_dx_icd_map is required.")

    # Merge icd_occurrences with disease_dx_icd_map on 'icd' and 'flag'
    dx_icd = pd.merge(icd_occurrences, disease_dx_icd_map[['disease_id', 'icd', 'flag']], on=['icd', 'flag'])

    # Calculate unique ages for each person and disease
    dx_icd_unique_ages = dx_icd.groupby(['person_id', 'disease_id'])['occurrence_age'].nunique().reset_index()
    dx_icd_unique_ages.rename(columns={'occurrence_age': 'uniq_ages'}, inplace=True)

    # Assign dx_status based on 'uniq_ages'
    dx_icd_unique_ages['dx_status'] = np.where(dx_icd_unique_ages['uniq_ages'] >= min_unique_ages, 1, -1)

    # Prepare a DataFrame to merge with
    all_combinations = demos[['person_id']].assign(key=1).merge(disease_dx_icd_map[['disease_id']].assign(key=1),
                                                                on='key').drop('key', axis=1)

    # Merge to assign dx_status to all combinations of person_id and disease_id
    dx_status = pd.merge(all_combinations, dx_icd_unique_ages[['person_id', 'disease_id', 'dx_status']],
                         on=['person_id', 'disease_id'], how='left')

    # Fill NaN values in dx_status with 0
    dx_status['dx_status'].fillna(0, inplace=True)
    dx_status['dx_status'] = dx_status['dx_status'].astype(int)

    return dx_status
