import patsy


def check_demos(demos, method='prevalence'):
    assert isinstance(demos, pd.DataFrame), "demos must be a pandas DataFrame"
    required_cols = ['person_id']
    excluded_cols = ['phecode', 'w', 'disease_id', 'score']
    if method == 'cox':
        required_cols.extend(['first_age', 'last_age'])
        excluded_cols.append('occurrence_age')
    elif method == 'loglinear':
        excluded_cols.append('num_occurrences')

    assert all(col in demos.columns for col in required_cols), f"Missing required columns: {required_cols}"
    assert not any(col in demos.columns for col in excluded_cols), f"DataFrame contains unexpected columns: {excluded_cols}"
    assert demos['person_id'].is_unique, "person_id must be unique"
    if method == 'cox':
        assert demos['first_age'].dtype == 'float' and demos['last_age'].dtype == 'float', "first_age and last_age must be numeric"
        assert (demos['first_age'] >= 0).all() and (demos['last_age'] >= 0).all(), "first_age and last_age must be non-negative"


def check_dx_icd(dx_icd, null_ok):
    if dx_icd is not None:
        assert isinstance(dx_icd, pd.DataFrame), "dx_icd must be a pandas DataFrame or None"
        if not null_ok:
            required_cols = ['disease_id', 'icd', 'flag']
        else:
            required_cols = ['icd', 'flag']

        assert all(col in dx_icd.columns for col in required_cols), f"Missing required columns: {required_cols}"
        assert 'person_id' not in dx_icd.columns, "DataFrame should not contain 'person_id' column"
        assert pd.api.types.is_string_dtype(dx_icd['icd']), "icd column must contain strings"
        assert dx_icd.columns.is_unique, "All column names must be unique"
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
    # Check if icd_phecode_map is a pandas DataFrame
    if not isinstance(icd_phecode_map, pd.DataFrame):
        raise ValueError("icd_phecode_map must be a pandas DataFrame")

    # Required columns
    required_columns = ['phecode', 'icd', 'flag']

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in icd_phecode_map.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for unique column names
    if icd_phecode_map.columns.duplicated().any():
        raise ValueError("Column names in icd_phecode_map must be unique")

    # Check data type of 'icd' and 'phecode' columns
    if icd_phecode_map['icd'].dtype != object:
        raise ValueError("Column 'icd' must contain strings")
    if icd_phecode_map['phecode'].dtype != object:
        raise ValueError("Column 'phecode' must contain strings")

    # Check for duplicates
    if icd_phecode_map.duplicated().any():
        raise ValueError("icd_phecode_map contains duplicate rows")



def check_icd_occurrences(icd_occurrences, cols=['person_id', 'icd', 'flag']):
    """
    Validates the ICD Occurrences DataFrame.

    Parameters:
    - icd_occurrences: pandas DataFrame, the ICD occurrences data.
    - cols: list, the required columns in the ICD occurrences data.

    The function checks:
    - icd_occurrences is a pandas DataFrame.
    - It contains specific required columns and doesn't include disallowed columns.
    - The 'icd' column must contain strings.
    """
    # Check if icd_occurrences is a pandas DataFrame
    if not isinstance(icd_occurrences, pd.DataFrame):
        raise ValueError("icd_occurrences must be a pandas DataFrame")

    # Check for required columns
    missing_cols = [col for col in cols if col not in icd_occurrences.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for unique column names
    if icd_occurrences.columns.duplicated().any():
        raise ValueError("Column names in icd_occurrences must be unique")

    # Check for disallowed columns
    disallowed_columns = ['phecode', 'disease_id']
    included_disallowed_cols = [col for col in disallowed_columns if col in icd_occurrences.columns]
    if included_disallowed_cols:
        raise ValueError(f"icd_occurrences should not include columns: {included_disallowed_cols}")

    # Check data type of 'icd' column
    if icd_occurrences['icd'].dtype != object:
        raise ValueError("Column 'icd' must contain strings")



def check_phecode_occurrences(phecode_occurrences, demos, method='prevalence'):
    """
    Validates the phecode occurrences DataFrame based on the analysis method.

    Parameters:
    - phecode_occurrences: pandas DataFrame, the phecode occurrences data.
    - demos: pandas DataFrame, demographic information of individuals in the cohort.
    - method: str, the analysis method that affects validation rules.
    """
    # Define required columns based on method
    cols = ['person_id', 'phecode']
    if method == 'cox':
        cols.extend(['occurrence_age'])
    elif method == 'loglinear':
        cols.extend(['num_occurrences'])

    # Check for required columns
    missing_cols = [col for col in cols if col not in phecode_occurrences.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for method '{method}': {missing_cols}")

    # Ensure column names are unique
    if phecode_occurrences.columns.duplicated().any():
        raise ValueError("Column names in phecode_occurrences must be unique")

    # Disallowed columns check
    disallowed_columns = ['w', 'disease_id']
    included_disallowed_cols = [col for col in disallowed_columns if col in phecode_occurrences.columns]
    if included_disallowed_cols:
        raise ValueError(f"phecode_occurrences should not include columns: {included_disallowed_cols}")

    # Numeric columns validation
    if method in ['cox', 'loglinear']:
        numeric_column = 'occurrence_age' if method == 'cox' else 'num_occurrences'
        if not pd.api.types.is_numeric_dtype(phecode_occurrences[numeric_column]):
            raise ValueError(f"Column '{numeric_column}' must be numeric for method '{method}'")

    # Validate 'person_id' in demos
    if not set(phecode_occurrences['person_id']).issubset(set(demos['person_id'])):
        raise ValueError("All 'person_id' values in phecode_occurrences must exist in demos")

    # Validate 'phecode' column is character type
    if not pd.api.types.is_string_dtype(phecode_occurrences['phecode']):
        raise ValueError("Column 'phecode' must contain strings")


def check_weights(weights):
    """
    Validates the weights DataFrame to ensure it meets the expected structure and types.

    Parameters:
    - weights: pandas DataFrame, the weights data to validate.
    """
    # Check for required columns
    required_columns = ['person_id', 'phecode', 'w']
    missing_cols = [col for col in required_columns if col not in weights.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure column names are unique and exclude disallowed columns
    if weights.columns.duplicated().any():
        raise ValueError("Column names in weights must be unique")
    if 'disease_id' in weights.columns:
        raise ValueError("'disease_id' should not be included in weights")

    # Check for duplicates based on 'person_id' and 'phecode'
    if weights.duplicated(subset=['person_id', 'phecode']).any():
        raise ValueError("Duplicates found based on 'person_id' and 'phecode'")

    # Validate 'phecode' column is character type
    if not pd.api.types.is_string_dtype(weights['phecode']):
        raise ValueError("Column 'phecode' must contain strings")

    # Validate 'w' column is numeric and finite
    if not pd.api.types.is_numeric_dtype(weights['w']) or weights['w'].isnull().any():
        raise ValueError("Column 'w' must be numeric and finite")


def check_disease_phecode_map(disease_phecode_map):
    """
    Validates the disease_phecode_map DataFrame to ensure it meets the expected structure and types.

    Parameters:
    - disease_phecode_map: pandas DataFrame, the disease to phecode mapping data to validate.
    """
    # Check for required columns
    required_columns = ['disease_id', 'phecode']
    missing_cols = [col for col in required_columns if col not in disease_phecode_map.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure column names are unique and exclude disallowed columns
    if disease_phecode_map.columns.duplicated().any():
        raise ValueError("Column names in disease_phecode_map must be unique")
    disallowed_cols = ['id', 'person_id', 'w']
    if any(col in disease_phecode_map.columns for col in disallowed_cols):
        raise ValueError(f"Columns {disallowed_cols} should not be included in disease_phecode_map")

    # Validate 'phecode' column is character type
    if not pd.api.types.is_string_dtype(disease_phecode_map['phecode']):
        raise ValueError("Column 'phecode' must contain strings")

    # Check for duplicates based on 'disease_id' and 'phecode'
    if disease_phecode_map.duplicated(subset=['disease_id', 'phecode']).any():
        raise ValueError("Duplicates found based on 'disease_id' and 'phecode'")


import pandas as pd

def check_scores(scores):
    """
    Validates the scores DataFrame to ensure it meets the expected structure and types.

    Parameters:
    - scores: pandas DataFrame, containing the scores data to validate.
    """
    # Check for required columns
    required_columns = ['person_id', 'disease_id', 'score']
    missing_cols = [col for col in required_columns if col not in scores.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure column names are unique
    if scores.columns.duplicated().any():
        raise ValueError("Column names in scores must be unique")

    # Validate 'score' column is numeric and contains finite values
    if not pd.api.types.is_numeric_dtype(scores['score']):
        raise ValueError("Column 'score' must contain numeric data")
    if not scores['score'].apply(pd.api.types.is_float).all():
        raise ValueError("Column 'score' must contain finite numbers")

    # Check for duplicates based on 'person_id' and 'disease_id'
    if scores.duplicated(subset=['person_id', 'disease_id']).any():
        raise ValueError("Duplicates found based on 'person_id' and 'disease_id'")


import pandas as pd

def check_genotypes(genotypes):
    """
    Validates the genotypes DataFrame to ensure it meets the expected structure and types.

    Parameters:
    - genotypes: pandas DataFrame, containing the genotypes data to validate.
    """
    # Assuming genotypes is a pandas DataFrame
    if not isinstance(genotypes, pd.DataFrame):
        raise TypeError("Genotypes must be a pandas DataFrame")

    # Check for unique row indices
    if genotypes.index.duplicated().any():
        raise ValueError("Row names (indices) in genotypes must be unique")

    # Check for unique column names
    if genotypes.columns.duplicated().any():
        raise ValueError("Column names in genotypes must be unique")

    # Ensure column names do not include 'score'
    if 'score' in genotypes.columns:
        raise ValueError("Column names in genotypes should not include 'score'")


import pandas as pd


def check_disease_variant_map(disease_variant_map, scores, genotypes):
    """
    Validates the disease variant mapping DataFrame.

    Parameters:
    - disease_variant_map: pandas DataFrame, the mapping between disease IDs and variant IDs.
    - scores: pandas DataFrame, containing scores for diseases.
    - genotypes: pandas DataFrame or BEDMatrix, containing genotype information.
    """
    # Check for a pandas DataFrame
    if not isinstance(disease_variant_map, pd.DataFrame):
        raise TypeError("disease_variant_map must be a pandas DataFrame")

    # Check for unique column names and required columns
    if disease_variant_map.columns.duplicated().any():
        raise ValueError("Column names in disease_variant_map must be unique")
    if not {'disease_id', 'variant_id'}.issubset(disease_variant_map.columns):
        raise ValueError("disease_variant_map must include 'disease_id' and 'variant_id' columns")
    if disease_variant_map.duplicated(subset=['disease_id', 'variant_id']).any():
        raise ValueError("Duplicates found in 'disease_id' and 'variant_id' combinations")

    # Check if disease_id in disease_variant_map is a subset of those in scores
    if not set(disease_variant_map['disease_id']).issubset(set(scores['disease_id'])):
        raise ValueError("All disease_id values in disease_variant_map must be present in scores")

    # Check if variant_id in disease_variant_map is a subset of genotype columns (assuming genotypes as DataFrame for simplicity)
    if isinstance(genotypes, pd.DataFrame):  # Adjust as needed for BEDMatrix equivalent in Python
        if not set(disease_variant_map['variant_id']).issubset(genotypes.columns):
            raise ValueError("All variant_id values in disease_variant_map must be column names in genotypes")
    else:
        raise TypeError("genotypes should be a pandas DataFrame or a BEDMatrix equivalent")



def check_lm_formula(lm_formula, demos):
    """
    Validates the linear model formula.

    Parameters:
    - lm_formula: string, the formula for the linear model.
    - demos: pandas DataFrame, containing demographic and other relevant data for the model.
    """
    # Parsing the formula to check its validity
    try:
        term_names = patsy.ModelDesc.from_formula(lm_formula).term_names
    except Exception as e:
        raise ValueError(f"Error parsing formula: {e}")

    # Ensuring the terms in the formula are present in demos columns and
    # do not include disallowed names
    disallowed_names = {'score', 'allele_count', 'person_id', 'disease_id'}
    missing_terms = [term for term in term_names if term not in demos.columns]
    disallowed_terms = [term for term in term_names if term in disallowed_names]

    if missing_terms:
        raise ValueError(f"Formula terms not found in demos: {missing_terms}")
    if disallowed_terms:
        raise ValueError(f"Formula contains disallowed terms: {disallowed_terms}")

    # Check if the formula contains a dependent variable, which is not allowed
    # This example assumes that a dependent variable would be indicated with a '~'
    # which is standard in patsy's formula syntax
    if '~' in lm_formula or '1' not in term_names:
        raise ValueError("The formula contains a dependent variable, which is not allowed.")



def check_lm_input(lm_input):
    """
    Validates the linear model input DataFrame.

    Parameters:
    - lm_input: pandas DataFrame, expected to contain 'score' and 'allele_count' columns.
    """
    # Ensure lm_input is a pandas DataFrame
    if not isinstance(lm_input, pd.DataFrame):
        raise ValueError("lm_input must be a pandas DataFrame.")

    # Check required columns
    required_columns = {'score', 'allele_count'}
    missing_columns = required_columns - set(lm_input.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check allele_count is numeric and finite
    if not pd.api.types.is_numeric_dtype(lm_input['allele_count']):
        raise ValueError("allele_count must be numeric.")
    if not np.all(np.isfinite(lm_input['allele_count'])):
        raise ValueError("allele_count must contain finite values.")

    # Check allele_count values are within the expected set (0, 1, 2)
    if not lm_input['allele_count'].isin([0, 1, 2]).all():
        raise ValueError("allele_count must contain values in [0, 1, 2].")



def get_allele_counts(lm_input):
    """
    Calculates counts of wild-type, heterozygous, and homozygous alleles.

    Parameters:
    - lm_input: pandas DataFrame, expected to contain an 'allele_count' column.

    Returns:
    A pandas DataFrame with counts for wild-type, heterozygous, and homozygous alleles.
    """
    # Ensure lm_input is a pandas DataFrame
    if not isinstance(lm_input, pd.DataFrame):
        raise ValueError("lm_input must be a pandas DataFrame.")

    # Calculate allele counts
    n_total = len(lm_input)
    n_wt = (lm_input['allele_count'] == 0).sum()
    n_het = (lm_input['allele_count'] == 1).sum()
    n_hom = (lm_input['allele_count'] == 2).sum()

    # Create a DataFrame to store the counts
    d_counts = pd.DataFrame({
        'n_total': [n_total],
        'n_wt': [n_wt],
        'n_het': [n_het],
        'n_hom': [n_hom]
    })

    return d_counts


def report_subset_assertions(x, choices):
    """
    Checks if all elements in x are present within choices and raises an error if not.

    Parameters:
    - x: Iterable, elements to check.
    - choices: Iterable, collection to check against.
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


import patsy

def check_method_formula(method_formula, demos):
    """
    Checks if the provided method formula is valid based on the demos DataFrame.

    Parameters:
    - method_formula: str or patsy.Formula object, the formula to check.
    - demos: pandas.DataFrame, the DataFrame against which to validate the formula.
    """
    # Convert the formula to a patsy Formula object if it's a string
    if isinstance(method_formula, str):
        formula = patsy.ModelDesc.from_formula(method_formula)
    else:
        formula = method_formula

    # Extract variable names from the formula
    formula_vars = {str(term) for term in formula.rhs_termlist}

    # Check if formula variables are in the DataFrame columns
    missing_vars = formula_vars.difference(demos.columns)
    if missing_vars:
        raise ValueError(f"The formula contains variables not in the DataFrame: {missing_vars}")

    # Check for forbidden variables
    forbidden_vars = {'dx_status', 'person_id', 'phecode'}
    if not forbidden_vars.isdisjoint(formula_vars):
        raise ValueError(f"The formula should not include variables: {forbidden_vars & formula_vars}")

    # Check if the formula contains a dependent variable
    if formula.lhs_termlist:
        raise ValueError("The formula contains a dependent variable, which is not allowed.")










