import sys
import polars as pl
from src.PheRS import utils


def get_dx_status(platform='aou', demos_path=None, icd_occurrences=None, min_unique_dates=2, disease_dx_icd_map=None):
    """
    Calculate dx_status for each person and disease based on ICD occurrences.

    :param demos: DataFrame containing demographic information with 'person_id'.
    :param icd_occurrences: DataFrame containing ICD code occurrences with 'person_id', 'icd', 'flag', 'occurrence_age'.
    :param min_unique_ages: Minimum number of unique ages at which a diagnosis must be recorded to set dx_status to 1.
    :param disease_dx_icd_map: DataFrame mapping disease IDs to ICD codes with 'disease_id', 'icd', 'flag'.
    :return: DataFrame with 'person_id', 'disease_id', 'dx_status'.
    """

    if platform == "aou":
        demos = utils.get_allofus_demo()
    elif platform == "custom":
        if demos_path is not None:
            print("\033[1mLoading user's demography data from file...")
            demos = pl.read_csv(demos_path)
        else:
            print("demos_path is required for custom platform.")
            sys.exit(0)
    else:
        print("Invalid platform. Parameter platform only accepts \"aou\" (All of Us) or \"custom\".")
        sys.exit(0)

    if icd_occurrences is None:
        print("icd_occurrences is required.")
        sys.exit(0)

    # Validate inputs (Placeholder for actual checks)
    utils.check_demos(demos)
    utils.check_icd_occurrences(icd_occurrences)
    # Check for numeric 'occurrence_date' in icd_occurrences and positive 'min_unique_dates'
    assert icd_occurrences['date'].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64], \
        "occurrence_age must be numeric"
    assert min_unique_dates > 0, "min_unique_ages must be positive"

    if disease_dx_icd_map is None:
        raise ValueError("disease_dx_icd_map is required.")

    # Merge icd_occurrences with disease_dx_icd_map on 'icd' and 'flag'
    dx_icd = icd_occurrences.join(disease_dx_icd_map, on=['icd', 'flag'], how='inner')

    # Calculate unique ages for each person and disease
    dx_icd_unique_dates = dx_icd.groupby(['person_id', 'disease_id']).agg(pl.count('date').alias('uniq_dates'))

    # Assign dx_status based on 'uniq_ages'
    dx_icd_unique_dates = dx_icd_unique_dates.with_columns(
        (pl.when(pl.col('uniq_ages') >= min_unique_dates).then(pl.lit(1)).otherwise(pl.lit(-1))).alias('dx_status')
    )

    # Create all combinations of person_id and disease_id
    all_combinations = pl.DataFrame({
        'person_id': demos['person_id'].to_list(),
        'disease_id': disease_dx_icd_map['disease_id'].unique().to_list()
    }).join(pl.DataFrame(disease_dx_icd_map['disease_id'].unique().repeat(len(demos['person_id']))), on='person_id', how='cross')

    # Merge to assign dx_status to all combinations of person_id and disease_id
    dx_status = all_combinations.join(dx_icd_unique_dates, on=['person_id', 'disease_id'], how='left')

    # Fill NaN values in dx_status with 0
    dx_status = dx_status.fill_null(0).with_columns(pl.col('dx_status').cast(pl.Int32))

    return dx_status

