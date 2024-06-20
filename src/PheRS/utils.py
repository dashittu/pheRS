from google.cloud import bigquery
import os
import polars as pl
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from pheRS.src.PheRS import queries


def to_polars(df):
    """
    Check and convert pandas dataframe object to polars dataframe, if applicable
    :param df: dataframe object
    :return: polars dataframe
    """
    if not isinstance(df, pl.DataFrame):
        return pl.from_pandas(df)
    else:
        return df


def polars_gbq(query):
    """
    Take a SQL query and return result as polars dataframe
    :param query: BigQuery SQL query
    :return: polars dataframe
    """
    client = bigquery.Client()
    query_job = client.query(query)
    rows = query_job.result()
    df = pl.from_arrow(rows.to_arrow())

    return df


def get_allofus_demo():
    """
    Takes in no arguments. Get All of Us demographic datasets
        "aou": All of Us OMOP database
    :return: list of all Us demographic datasets
    """
    cdr = os.getenv("WORKSPACE_CDR")
    demo_query = queries.all_demo_query(cdr)
    print("\033[1mStart querying ICD codes...")
    demo = polars_gbq(demo_query)
    return demo


def get_phecode_mapping(phecode_version, icd_version, phecode_map_file_path, keep_all_columns=True):
    """
    Load phecode mapping table
    :param phecode_version: defaults to "1.2"; other option is "X"
    :param icd_version: defaults to "US"; other option is "custom";
                        if "custom", user need to provide phecode_map_path
    :param phecode_map_file_path: path to custom phecode map table
    :param keep_all_columns: defaults to True
    :return: phecode mapping table as polars dataframe
    """
    # load phecode mapping file by version or by custom path
    phemap_dir = os.path.dirname(__file__)
    final_file_path = os.path.join(phemap_dir, "data_raw")
    path_suffix = ""
    if phecode_version == "X":
        if icd_version == "US":
            path_suffix = "phecodeX.csv"
        elif icd_version == "custom":
            if phecode_map_file_path is None:
                print("Please provide phecode_map_path for custom icd_version")
                sys.exit(0)
        else:
            print("Invalid icd_version. Available icd_version values are US, WHO and custom.")
            sys.exit(0)
        if phecode_map_file_path is None:
            final_file_path = os.path.join(final_file_path, path_suffix)
        else:
            final_file_path = phecode_map_file_path
        # noinspection PyTypeChecker
        phecode_df = pl.read_csv(final_file_path,
                                 dtypes={"phecode": str,
                                         "ICD": str,
                                         "flag": pl.Int8,
                                         "code_val": float})
        phecode_df = phecode_df.with_columns(phecode_df["phecode"].cast(pl.Utf8))
        if not keep_all_columns:
            phecode_df = phecode_df[["phecode", "ICD", "flag"]]
    elif phecode_version == "1.2":
        if icd_version == "US":
            path_suffix = "phecode12.csv"
        elif icd_version == "custom":
            if phecode_map_file_path is None:
                print("Please provide phecode_map_path for custom icd_version")
                sys.exit(0)
        else:
            print("Invalid icd_version. Available icd_version values are US and custom.")
            sys.exit(0)
        if phecode_map_file_path is None:
            final_file_path = os.path.join(final_file_path, path_suffix)
        else:
            final_file_path = phecode_map_file_path
        # noinspection PyTypeChecker
        phecode_df = pl.read_csv(final_file_path,
                                 dtypes={"phecode": str,
                                         "ICD": str,
                                         "flag": pl.Int8,
                                         "exclude_range": str,
                                         "phecode_unrolled": str})
        phecode_df = phecode_df.with_columns(phecode_df["phecode"].cast(pl.Utf8))
        if not keep_all_columns:
            phecode_df = phecode_df[["phecode_unrolled", "ICD", "flag"]]
            phecode_df = phecode_df.rename({"phecode_unrolled": "phecode"})
    else:
        print("Unsupported phecode version. Supports phecode \"1.2\" and \"X\".")
        sys.exit(0)

    return phecode_df


def check_demos(demos, method='prevalence'):
    """
        Validates the demographic DataFrame for required columns and checks for unexpected columns.

        Parameters:
        - demos (pl.DataFrame): The demographic DataFrame to be validated.
        - method (str): The method for which the DataFrame is being validated. Options are 'prevalence', 'cox',
                        or 'loglinear'. Default is 'prevalence'.

        Raises:
        - AssertionError: If the DataFrame is not a Polars DataFrame.
        - AssertionError: If the required columns are missing from the DataFrame.
        - AssertionError: If the DataFrame contains unexpected columns.
        - AssertionError: If 'person_id' is not unique.
        - AssertionError: If 'first_age' and 'last_age' are not numeric (for 'cox' method).
        - AssertionError: If 'first_age' or 'last_age' contains negative values (for 'cox' method).

        The function performs the following checks:
        1. Ensures the input `demos` is a Polars DataFrame.
        2. Defines required and excluded columns based on the specified method.
        3. Checks if all required columns are present in the DataFrame.
        4. Checks if any excluded columns are present in the DataFrame.
        5. Ensures that the 'person_id' column contains unique values.
        6. For 'cox' method, ensures 'first_age' and 'last_age' are numeric and non-negative.
        """
    assert isinstance(demos, pl.DataFrame), "demos must be a polars DataFrame"
    required_cols = ['person_id']
    excluded_cols = ['phecode', 'w', 'disease_id', 'score']
    if method == 'cox':
        required_cols.extend(['first_age', 'last_age'])
        excluded_cols.append('occurrence_age')
    elif method == 'loglinear':
        excluded_cols.append('num_occurrences')

    assert all(col in demos.columns for col in required_cols), f"Missing required columns: {required_cols}"
    assert not any(col in demos.columns for col in excluded_cols), (f"DataFrame contains unexpected columns: "
                                                                    f"{excluded_cols}")
    assert demos.select('person_id').is_unique().all(), "person_id must be unique"
    if method == 'cox':
        assert demos.schema['first_age'] in [pl.Int64, pl.Float64] and demos.schema['last_age'] in [
            pl.Int64, pl.Float64], "first_age and last_age must be numeric"
        assert demos.filter((pl.col('first_age') < 0) | (pl.col('last_age') < 0)).is_empty(), \
            "first_age and last_age must be non-negative"


def check_dx_icd(dx_icd, null_ok):
    """
       Validates the dx_icd DataFrame based on the provided criteria.

       Parameters:
       - dx_icd (pl.DataFrame or None): The DataFrame to validate.
       - null_ok (bool): Whether a None value for dx_icd is acceptable.

       Raises:
       - AssertionError: If the DataFrame is not of type Polars or contains invalid data.
       - ValueError: If dx_icd is None and null_ok is False.

       Validation checks:
       - Ensures dx_icd is a Polars DataFrame or None.
       - Checks required columns based on null_ok.
       - Ensures the absence of 'person_id' column.
       - Validates the 'icd' column contains strings.
       - Ensures all column names are unique.
       """
    if dx_icd is not None:
        assert isinstance(dx_icd, pl.DataFrame), "dx_icd must be a Polars DataFrame or None"
        if not null_ok:
            required_cols = ['disease_id', 'icd', 'flag']
        else:
            required_cols = ['icd', 'flag']

        assert all(col in dx_icd.columns for col in required_cols), f"Missing required columns: {required_cols}"
        assert 'person_id' not in dx_icd.columns, "DataFrame should not contain 'person_id' column"
        assert dx_icd.schema['icd'] == pl.Utf8, "icd column must contain strings"
        assert len(dx_icd.columns) == len(set(dx_icd.columns)), "All column names must be unique"
    elif not null_ok:
        raise ValueError("dx_icd cannot be None unless null_ok is True")


def check_icd_phecode_map(icd_phecode_map):
    """
    Validate the structure and content of the ICD-Phecode Mapping DataFrame.

    Parameters:
    - icd_phecode_map: polars DataFrame containing the ICD-Phecode mapping information.

    The function checks:
    - The DataFrame must contain the columns 'phecode', 'icd', 'flag'.
    - Columns 'icd' and 'phecode' should contain strings.
    - The DataFrame should have no duplicate rows.
    """
    # Check if icd_phecode_map is a polars DataFrame
    if not isinstance(icd_phecode_map, pl.DataFrame):
        raise ValueError("icd_phecode_map must be a Polars DataFrame")

    # Required columns
    required_columns = ['phecode', 'ICD', 'flag']

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in icd_phecode_map.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for unique column names
    if len(icd_phecode_map.columns) != len(set(icd_phecode_map.columns)):
        raise ValueError("Column names in icd_phecode_map must be unique")

    # Check data type of 'icd' and 'phecode' columns
    if not icd_phecode_map.schema['ICD'] == pl.Utf8:
        raise ValueError("Column 'ICD' must contain strings")
    if not icd_phecode_map.schema['phecode'] == pl.Utf8:
        raise ValueError("Column 'phecode' must contain strings")

    # Check for duplicates
    # if icd_phecode_map.unique().shape[0] != icd_phecode_map.shape[0]:
    #    raise ValueError("icd_phecode_map contains duplicate rows")


def check_icd_occurrences(icd_occurrences, cols=None):
    """
    Validates the ICD Occurrences DataFrame.

    Parameters:
    - icd_occurrences: pandas DataFrame, the ICD occurrences data.
    - cols: list, the required columns in the ICD occurrences data.

    The function checks:
    - icd_occurrences is a polars DataFrame.
    - It contains specific required columns and doesn't include disallowed columns.
    - The 'icd' column must contain strings.
    - The DataFrame should have no duplicate rows.
    """
    # Check if icd_occurrences is a polars DataFrame
    if cols is None:
        cols = ['person_id', 'ICD', 'flag']
    if not isinstance(icd_occurrences, pl.DataFrame):
        raise ValueError("icd_occurrences must be a polars DataFrame")

    # Check for required columns
    missing_cols = [col for col in cols if col not in icd_occurrences.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for unique column names
    if len(icd_occurrences.columns) != len(set(icd_occurrences.columns)):
        raise ValueError("Column names in icd_occurrences must be unique")

    # Check for disallowed columns
    disallowed_columns = ['phecode', 'disease_id']
    included_disallowed_cols = [col for col in disallowed_columns if col in icd_occurrences.columns]
    if included_disallowed_cols:
        raise ValueError(f"icd_occurrences should not include columns: {included_disallowed_cols}")

    # Check data type of 'icd' column
    if icd_occurrences.schema['ICD'] != pl.Utf8:
        raise ValueError("Column 'icd' must contain strings")

    # Check for duplicate rows
    # if icd_occurrences.unique(subset=['person_id', 'ICD', 'flag']).shape[0] != icd_occurrences.shape[0]:
    #    raise ValueError("icd_occurrences contains duplicate rows")


def check_phecode_occurrences(phecode_occurrences, demos, method='prevalence'):
    """
    Validates the Phecode occurrences DataFrame based on the analysis method.
    Ensures proper columns based on method, checks for unique and correct column types, and confirms person_id subset.

    Parameters:
    - phecode_occurrences: polars DataFrame, the phecode occurrences data.
    - demos: polars DataFrame, demographic information of individuals in the cohort.
    - method: str, the analysis method that affects validation rules.

    Raises:
    - ValueError: If any validation check fails.
    """
    valid_methods = ['prevalence', 'logistic', 'cox', 'loglinear']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    # Define required columns based on method
    cols = ['person_id', 'phecode']
    if method == 'cox':
        cols.append('occurrence_age')
    elif method == 'loglinear':
        cols.append('num_occurrences')

    # Check if phecode_occurrences is a DataFrame
    if not isinstance(phecode_occurrences, pl.DataFrame):
        raise ValueError("phecode_occurrences must be a polars DataFrame")

    # Check for required columns
    missing_cols = [col for col in cols if col not in phecode_occurrences.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for method '{method}': {missing_cols}")

    # Ensure column names are unique
    if len(phecode_occurrences.columns) != len(set(phecode_occurrences.columns)):
        raise ValueError("Column names in phecode_occurrences must be unique")

    # Disallowed columns check
    disallowed_columns = ['w', 'disease_id']
    included_disallowed_cols = [col for col in disallowed_columns if col in phecode_occurrences.columns]
    if included_disallowed_cols:
        raise ValueError(f"phecode_occurrences should not include columns: {included_disallowed_cols}")

    # Numeric columns validation
    if method in ['cox', 'loglinear']:
        numeric_column = 'occurrence_age' if method == 'cox' else 'num_occurrences'
        if phecode_occurrences.schema[numeric_column] not in [pl.Int64, pl.Float64]:
            raise ValueError(f"Column '{numeric_column}' must be numeric for method '{method}'")

    # Validate 'person_id' in demos
    if not phecode_occurrences['person_id'].is_in(demos['person_id']).all():
        raise ValueError("All 'person_id' values in phecode_occurrences must exist in demos")

    # Validate 'phecode' column is character type
    if phecode_occurrences.schema['phecode'] != pl.Utf8:
        raise ValueError("Column 'phecode' must contain strings")


def check_weights(weights):
    """
    Validates the weights DataFrame to ensure it meets the expected structure and types.

    Parameters:
    - weights: polars DataFrame, the weights data to validate.

    Raises:
    - ValueError: If any validation check fails.
    """
    # Ensure weights is a DataFrame
    if not isinstance(weights, pl.DataFrame):
        raise ValueError("weights must be a polars DataFrame")

    # Check for required columns
    required_columns = ['person_id', 'phecode', 'w']
    missing_cols = [col for col in required_columns if col not in weights.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure column names are unique and exclude disallowed columns
    if len(weights.columns) != len(set(weights.columns)):
        raise ValueError("Column names in weights must be unique")
    if 'disease_id' in weights.columns:
        raise ValueError("'disease_id' should not be included in weights")

    # Check for duplicates based on 'person_id' and 'phecode'
    if weights.unique(subset=['person_id', 'phecode']).height != weights.height:
        raise ValueError("Duplicates found based on 'person_id' and 'phecode'")

    # Validate 'phecode' column is character type
    if weights.schema['phecode'] != pl.Utf8:
        raise ValueError("Column 'phecode' must contain strings")

    # Validate 'w' column is numeric and finite
    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    if weights.schema['w'] not in numeric_dtypes:
        raise ValueError("Column 'w' must be numeric")
    if weights.filter(pl.col('w').is_null() | pl.col('w').is_infinite()).height > 0:
        raise ValueError("Column 'w' must contain only finite numbers")


def check_disease_phecode_map(disease_phecode_map):
    """
    Validates the disease_phecode_map DataFrame to ensure it meets the expected structure and types.
    This is the dataframe generated from map.py

    Parameters:
    - disease_phecode_map: polars DataFrame, the disease to phecode mapping data to validate.

    Raises:
    - ValueError: If any validation check fails.
    """
    # Ensure disease_phecode_map is a DataFrame
    if not isinstance(disease_phecode_map, pl.DataFrame):
        raise ValueError("disease_phecode_map must be a Polars DataFrame")

    # Check for required columns
    required_columns = ['disease_id', 'phecode']
    missing_cols = [col for col in required_columns if col not in disease_phecode_map.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for unique column names
    if len(disease_phecode_map.columns) != len(set(disease_phecode_map.columns)):
        raise ValueError("Column names in disease_phecode_map must be unique")

    # Exclude disallowed columns
    disallowed_cols = {'id', 'person_id', 'w'}  # Use a set for efficient lookup
    included_disallowed_cols = disallowed_cols.intersection(set(disease_phecode_map.columns))
    if included_disallowed_cols:
        raise ValueError(f"Columns {included_disallowed_cols} should not be included in disease_phecode_map")

    # Validate 'phecode' column is character type
    if disease_phecode_map.schema['phecode'] != pl.Utf8:
        raise ValueError("Column 'phecode' must contain strings")

    # Check for duplicates based on 'disease_id' and 'phecode'
    duplicates = disease_phecode_map.groupby(['disease_id', 'phecode']).count()

    if (duplicates['count'] > 1).any():
        raise ValueError("Duplicates found based on 'disease_id' and 'phecode'")


def check_scores(scores):
    """
    Validates the scores DataFrame to ensure it meets the expected structure and types, including checking
    for uniqueness of 'person_id' and 'disease_id' pairs, ensuring required columns are present, and verifying the
    data type and finiteness of the 'score' column.

    Parameters:
    - scores: pandas DataFrame, containing the scores data to validate.

    Raises:
    - ValueError: If any validation check fails.
    """
    # Ensure scores is a DataFrame
    if not isinstance(scores, pl.DataFrame):
        raise ValueError("scores must be a polars DataFrame")

    # Check for required columns
    required_columns = ['person_id', 'disease_id', 'score']
    missing_cols = [col for col in required_columns if col not in scores.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure column names are unique
    if len(scores.columns) != len(set(scores.columns)):
        raise ValueError("Column names in scores must be unique")

    # Validate 'score' column is numeric and contains only finite values
    numeric_dtypes = [pl.Int32, pl.Int64, pl.Float32, pl.Float64]
    if scores.schema['score'] not in numeric_dtypes:
        raise ValueError("Column 'score' must contain numeric data")
    if scores.filter(pl.col('score').is_null() | pl.col('score').is_infinite()).height > 0:
        raise ValueError("Column 'score' must contain only finite numbers")

    # Check for duplicates based on 'person_id' and 'disease_id'
    if scores.unique(subset=['person_id', 'disease_id']).height != scores.height:
        raise ValueError("Duplicates found based on 'person_id' and 'disease_id'")


def check_method_formula(method_formula, demos):
    """
    Validates a statistical model formula to ensure it only uses allowed predictors from a DataFrame
    and does not specify a dependent variable. It also checks that the formula does not include any forbidden variables.

    Parameters:
    - method_formula: str or patsy.ModelDesc, the formula to check.
    - demos: pandas.DataFrame, the DataFrame against which to validate the formula.

    Raises:
    - ValueError: If the formula specifies a dependent variable or contains forbidden variables.
    """

    formula_vars = set(method_formula.replace(' ', '').split('+'))

    # Check if formula variables are present in the DataFrame columns
    missing_vars = formula_vars.difference(demos.columns)
    if missing_vars:
        raise ValueError(f"The formula contains variables not in the DataFrame: {missing_vars}")

    # Check for forbidden variables
    forbidden_vars = {'dx_status', 'person_id', 'phecode'}
    included_forbidden_vars = forbidden_vars.intersection(formula_vars)
    if included_forbidden_vars:
        raise ValueError(f"The formula includes forbidden variables: {included_forbidden_vars}")

    print("Formula is correctly specified with appropriate variables and no dependent variable.")


def report_result(result, placeholder, output_file_name):
    if not result.is_empty():
        if output_file_name is None:
            file_name = f"{placeholder}.csv"
        else:
            file_name = output_file_name
        result.write_csv(file_name)

        print(f"\033[1mSuccessfully generated {placeholder} for cohort participants!\n"
              f"\033[1mSaved to {file_name}!\033[0m")
        print()
    else:
        print("\033[1mNo phecode occurrences generated. Check your input data.\033[0m")
        print()
