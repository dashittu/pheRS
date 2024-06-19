import sys
import os
import polars as pl
# noinspection PyUnresolvedReferences,PyProtectedMember
from src.PheRS import utils


def get_dx_status(platform='aou', demos_path=None, icd_occurrences_path=None, min_unique_ages=2,
                  disease_dx_icd_map=None, output_file_name=None):
    """
    Calculate dx_status for each person and disease based on ICD occurrences.

    :param demos_path: Path to the dataframe containing demographic information with 'person_id'.
    :param platform: Population to calculate the dx_status for.
    :param icd_occurrences_path: DataFrame containing ICD code occurrences with 'person_id', 'icd', 'flag', 'occurrence_age'.
    :param min_unique_ages: Minimum number of unique ages at which a diagnosis must be recorded to set dx_status to 1.
    :param disease_dx_icd_map: DataFrame mapping disease IDs to ICD codes with 'disease_id', 'ICD', 'flag'.
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

    if icd_occurrences_path is not None:
        icd_occurrences = pl.read_csv(icd_occurrences_path)
    else:
        print("icd_occurrences is required.")
        sys.exit(0)

    # Validate inputs (Placeholder for actual checks)
    utils.check_demos(demos)
    utils.check_icd_occurrences(icd_occurrences)
    # Check for numeric 'occurrence_age' in icd_occurrences and positive 'min_unique_ages'
    assert icd_occurrences['occurrence_age'].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64], \
        "occurrence_age must be numeric"
    assert min_unique_ages > 0, "min_unique_ages must be positive"

    if disease_dx_icd_map is None:
        disease_icd_dir = os.path.dirname(__file__)
        disease_icd_dir = os.path.join(disease_icd_dir, "data_raw/disease_dx_icd_map_omim.csv")
        disease_dx_icd_map = pl.read_csv(disease_icd_dir)
    else:
        disease_dx_icd_map = pl.read_csv(disease_dx_icd_map)

    # Merge icd_occurrences with disease_dx_icd_map on 'icd' and 'flag'
    dx_icd = icd_occurrences.join(disease_dx_icd_map, on=['ICD', 'flag'], how='inner')

    dx_icd = dx_icd.with_columns(
        pl.col('occurrence_age').round(0).alias('occurrence_age')
    )

    # Calculate unique ages for each person and disease
    dx_icd_unique_ages = dx_icd.group_by(['person_id', 'disease_id']).agg(
        pl.col('occurrence_age').n_unique().alias('uniq_ages'))

    # Assign dx_status based on 'uniq_ages'
    dx_icd_unique_ages = dx_icd_unique_ages.with_columns(
        (pl.when(pl.col('uniq_ages') >= min_unique_ages).then(pl.lit(1)).otherwise(pl.lit(-1))).alias('dx_status')
    )

    # Create all combinations of person_id and disease_id using a cross join
    person_ids_df = demos.select(pl.col('person_id').alias('person_id'))
    disease_ids_df = disease_dx_icd_map.select(pl.col('disease_id').alias('disease_id')).unique()

    # Add a constant column to both DataFrames for the cross join
    person_ids_df = person_ids_df.with_columns(pl.lit(1).alias('join_key'))
    disease_ids_df = disease_ids_df.with_columns(pl.lit(1).alias('join_key'))

    # Perform the cross join using the join_key
    all_combinations = person_ids_df.join(disease_ids_df, on='join_key', how='inner').drop('join_key')

    # Merge to assign dx_status to all combinations of person_id and disease_id
    dx_status = all_combinations.join(dx_icd_unique_ages, on=['person_id', 'disease_id'], how='left')

    # Fill NaN values in dx_status with 0
    dx_status = dx_status.fill_null(0).with_columns(pl.col('dx_status').cast(pl.Int32))

    # Report result
    utils.report_result(dx_status, placeholder='dx_status', output_file_name=output_file_name)
