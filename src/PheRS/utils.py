from patsy.desc import ModelDesc
from google.cloud import bigquery
import os
import polars as pl
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from PheRS import queries


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
    :param phecode_version: defaults to "X"; other option is "1.2"
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
        if not keep_all_columns:
            phecode_df = phecode_df[["phecode_unrolled", "ICD", "flag"]]
            phecode_df = phecode_df.rename({"phecode_unrolled": "phecode"})
    else:
        print("Unsupported phecode version. Supports phecode \"1.2\" and \"X\".")
        sys.exit(0)

    return phecode_df


def check_demos(demos, method='prevalence'):
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
    if dx_icd is not None:
        assert isinstance(dx_icd, pl.DataFrame), "dx_icd must be a Polars DataFrame or None"
        if not null_ok:
            required_cols = ['disease_id', 'icd', 'flag']
        else:
            required_cols = ['icd', 'flag']

        assert all(col in dx_icd.columns for col in required_cols), f"Missing required columns: {required_cols}"
        assert 'person_id' not in dx_icd.columns, "DataFrame should not contain 'person_id' column"
        assert dx_icd.schema['icd'].dtype == pl.Utf8, "icd column must contain strings"
        assert len(dx_icd.columns) == len(set(dx_icd.columns)), "All column names must be unique"
    elif not null_ok:
        raise ValueError("dx_icd cannot be None unless null_ok is True")


def check_icd_phecode_map(icd_phecode_map):
    """
    Validate the structure and content of the ICD-Phecode Mapping DataFrame.

    Parameters:
    - icd_phecode_map: pandas DataFrame containing the ICD-Phecode mapping information.

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
    - icd_occurrences is a pandas DataFrame.
    - It contains specific required columns and doesn't include disallowed columns.
    - The 'icd' column must contain strings.
    - The DataFrame should have no duplicate rows.
    """
    # Check if icd_occurrences is a polars DataFrame
    if cols is None:
        cols = ['person_id', 'ICD', 'flag']
    if not isinstance(icd_occurrences, pl.DataFrame):
        raise ValueError("icd_occurrences must be a pandas DataFrame")

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
    if icd_occurrences.unique(subset=['person_id', 'ICD', 'flag']).shape[0] != icd_occurrences.shape[0]:
        raise ValueError("icd_occurrences contains duplicate rows")


def check_phecode_occurrences(phecode_occurrences, demos, method='prevalence'):
    """
    Validates the Phecode occurrences DataFrame based on the analysis method.
    Ensures proper columns based on method, checks for unique and correct column types, and confirms person_id subset.

    Parameters:
    - phecode_occurrences: polars DataFrame, the phecode occurrences data.
    - demos: pandas DataFrame, demographic information of individuals in the cohort.
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
    if weights.schema['phecode'].dtype != pl.Utf8:
        raise ValueError("Column 'phecode' must contain strings")

    # Validate 'w' column is numeric and finite
    numeric_dtypes = [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    if weights.schema['w'].dtype not in numeric_dtypes:
        raise ValueError("Column 'w' must be numeric")
    if weights.filter(pl.col('w').is_null() | pl.col('w').is_infinite()).height > 0:
        raise ValueError("Column 'w' must contain only finite numbers")

    # Optionally, assert that all weights are non-negative (if applicable)
    # if weights.filter(pl.col('w') < 0).height() > 0:
    #         raise ValueError("Column 'w' must contain non-negative values only")


def check_disease_phecode_map(disease_phecode_map):
    """
    Validates the disease_phecode_map DataFrame to ensure it meets the expected structure and types.

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
    if disease_phecode_map.duplicated(subset=['disease_id', 'phecode']).sum() > 0:
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
    if scores.schema['score'].dtype not in numeric_dtypes:
        raise ValueError("Column 'score' must contain numeric data")
    if scores.filter(pl.col('score').is_null() | pl.col('score').is_infinite()).height > 0:
        raise ValueError("Column 'score' must contain only finite numbers")

    # Check for duplicates based on 'person_id' and 'disease_id'
    if scores.unique(subset=['person_id', 'disease_id']).height != scores.height:
        raise ValueError("Duplicates found based on 'person_id' and 'disease_id'")


"""def check_genotypes(genotypes):
"""
"""
    Validates the genotypes DataFrame to ensure it meets the expected structure and types.

    Parameters:
    - genotypes: pandas DataFrame, containing the genotypes data to validate.
"""
"""   
    # Assuming genotypes is a pandas DataFrame
#    if not isinstance(genotypes, pd.DataFrame):
#        raise TypeError("Genotypes must be a pandas DataFrame")

    # Check for unique row indices
#    if genotypes.index.duplicated().any():
#        raise ValueError("Row names (indices) in genotypes must be unique")

    # Check for unique column names
#    if genotypes.columns.duplicated().any():
#        raise ValueError("Column names in genotypes must be unique")

    # Ensure column names do not include 'score'
#    if 'score' in genotypes.columns:
#        raise ValueError("Column names in genotypes should not include 'score'")

"""
"""
def check_disease_variant_map(disease_variant_map, scores, genotypes):
"""
"""
    Validates the disease variant mapping DataFrame.

    Parameters:
    - disease_variant_map: pandas DataFrame, the mapping between disease IDs and variant IDs.
    - scores: pandas DataFrame, containing scores for diseases.
    - genotypes: pandas DataFrame or BEDMatrix, containing genotype information.
    """
"""
    # Check for a pandas DataFrame
#    if not isinstance(disease_variant_map, pd.DataFrame):
#        raise TypeError("disease_variant_map must be a pandas DataFrame")

    # Check for unique column names and required columns
#    if disease_variant_map.columns.duplicated().any():
#        raise ValueError("Column names in disease_variant_map must be unique")
#    if not {'disease_id', 'variant_id'}.issubset(disease_variant_map.columns):
#        raise ValueError("disease_variant_map must include 'disease_id' and 'variant_id' columns")
#    if disease_variant_map.duplicated(subset=['disease_id', 'variant_id']).any():
#        raise ValueError("Duplicates found in 'disease_id' and 'variant_id' combinations")

    # Check if disease_id in disease_variant_map is a subset of those in scores
#    if not set(disease_variant_map['disease_id']).issubset(set(scores['disease_id'])):
#        raise ValueError("All disease_id values in disease_variant_map must be present in scores")

    # Check if variant_id in disease_variant_map is a subset of genotype columns
    # (assuming genotypes as DataFrame for simplicity)
#    if isinstance(genotypes, pd.DataFrame):  # Adjust as needed for BEDMatrix equivalent in Python
#        if not set(disease_variant_map['variant_id']).issubset(genotypes.columns):
#            raise ValueError("All variant_id values in disease_variant_map must be column names in genotypes")
#    else:
#        raise TypeError("genotypes should be a pandas DataFrame or a BEDMatrix equivalent")
"""


def check_lm_formula(lm_formula, demos):
    """
    Validates the linear model formula to ensure that it only contains allowed variables from a polars DataFrame
    and does not specify a dependent variable, which is not appropriate in this context.

    Parameters:
    - lm_formula: string, the formula for the linear model.
    - demos: polars DataFrame, containing demographic and other relevant data for the model.

    Raises:
    - ValueError: If the formula is incorrectly specified or contains disallowed variables.
    """
    # Parsing the formula to check its validity and extract terms
    try:
        terms = ModelDesc.from_formula(lm_formula)
    except Exception as e:
        raise ValueError(f"Error parsing formula: {e}")

    # Extracting predictor terms from the formula (assuming the left side of ~ is empty which is required)
    term_names = [str(term) for factor in terms for term in factor.factors if str(term) != '1']

    # Ensure the terms in the formula are present in demos columns and do not include disallowed names
    disallowed_names = {'score', 'allele_count', 'person_id', 'disease_id'}
    missing_terms = [term for term in term_names if term not in demos.columns]
    disallowed_terms = [term for term in term_names if term in disallowed_names]

    if missing_terms:
        raise ValueError(f"Formula terms not found in demos: {missing_terms}")
    if disallowed_terms:
        raise ValueError(f"Formula contains disallowed terms: {disallowed_terms}")

    # Check if the formula implies a dependent variable by checking the structure of the parsed terms
    if terms.lhs_termlist:
        raise ValueError("The formula contains a dependent variable, which is not allowed.")


def check_lm_input(lm_input):
    """
    Validates the linear model input DataFrame.

    Parameters:
    - lm_input: pandas DataFrame, expected to contain 'score' and 'allele_count' columns.
    """
    # Ensure lm_input is a pandas DataFrame
    if not isinstance(lm_input, pl.DataFrame):
        raise ValueError("lm_input must be a polars DataFrame.")

    # Check required columns
    required_columns = {'score', 'allele_count'}
    missing_columns = required_columns - set(lm_input.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check allele_count is numeric and finite
    numeric_dtypes = [pl.Int32, pl.Int64, pl.Float32, pl.Float64]
    if lm_input.schema['allele_count'].dtype not in numeric_dtypes:
        raise ValueError("allele_count must be numeric.")
    if lm_input.filter(pl.col('allele_count').is_null() | pl.col('allele_count').is_infinite()).height > 0:
        raise ValueError("allele_count must contain finite values.")

    # Create a mask where False indicates values outside {0, 1, 2}
    invalid_mask = lm_input.select(pl.col('allele_count').is_in([0, 1, 2]).alias('valid')).filter(
        pl.col('valid') is False)

    # Check if any False values exist
    if invalid_mask.height > 0:
        raise ValueError("allele_count must contain values in [0, 1, 2].")


def get_allele_counts(lm_input):
    """
    Calculates counts of wild-type, heterozygous, and homozygous alleles from a pandas DataFrame
    containing allele count data. Ensures input validation and returns a DataFrame with the total count and
    counts for each allele type.

    Parameters:
    - lm_input: pandas DataFrame, expected to contain an 'allele_count' column.

    Returns:
    A pandas DataFrame with the following columns:
        - n_total: Total number of observations
        - n_wt: Number of wild-type alleles (allele_count == 0)
        - n_het: Number of heterozygous alleles (allele_count == 1)
        - n_hom: Number of homozygous alleles (allele_count == 2)

    Raises:
    - ValueError: If the input DataFrame does not contain the required 'allele_count' column.
    """
    # Check if 'allele_count' column exists in the DataFrame
    if 'allele_count' not in lm_input.columns:
        raise ValueError("The DataFrame must contain an 'allele_count' column.")

    # Calculate allele counts
    n_total = lm_input.height
    n_wt = (lm_input['allele_count'] == 0).height
    n_het = (lm_input['allele_count'] == 1).height
    n_hom = (lm_input['allele_count'] == 2).height

    # Create a DataFrame to store the counts
    d_counts = pl.DataFrame({
        'n_total': [n_total],
        'n_wt': [n_wt],
        'n_het': [n_het],
        'n_hom': [n_hom]
    })

    return d_counts


"""
def report_subset_assertions(x, choices):
"""
"""
    Checks if all elements in x are present within choices and raises an error if not.

    Parameters:
    - x: Iterable, elements to check.
    - choices: Iterable, collection to check against.
"""
"""
    # Convert iterables to sets for efficient membership checking
    x_set = set(x)
    choices_set = set(choices)

    # Check if x is a subset of choices
    if not x_set.issubset(choices_set):
        # Find elements in x that are not in choices
        missing_elements = x_set.difference(choices_set)
        msg = f"Elements {missing_elements} in x must be a subset of choices."
        raise ValueError(msg)
"""


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
