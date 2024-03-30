import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from lifelines import CoxPHFitter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed




def get_weights_prevalence(demos, phecode_occurrences, negative_weights):
    """
        Calculate weights based on the prevalence of each phecode in the given dataset.

        Parameters:
        - demos: A pandas DataFrame containing demographic information. It must include at least 'person_id'.
        - phecode_occurrences: A pandas DataFrame listing occurrences of phecodes for each person.
          It must include the columns 'person_id' and 'phecode'. Each row represents an occurrence
          of a phecode for a person.
        - negative_weights: A boolean flag indicating whether to allow negative weights for individuals
          without occurrences of a phecode. If False, weights for these individuals will be set to 0.

        Returns:
        A pandas DataFrame containing weights calculated for each combination of 'person_id' and 'phecode'.
        The DataFrame includes the columns 'person_id', 'phecode', 'pred', and 'w', where 'pred' is the
        prevalence of the phecode in the dataset, and 'w' is the calculated weight.
        """

    # Calculate prevalence (pred) for each phecode
    phecode_counts = phecode_occurrences.groupby('phecode')['person_id'].nunique().reset_index()
    phecode_counts['pred'] = phecode_counts['person_id'] / len(demos)
    phecode_counts.rename(columns={'person_id': 'count'}, inplace=True)

    # Prepare a dataframe with all combinations of person_id and phecode
    all_combinations = demos.assign(key=1).merge(phecode_counts.assign(key=1), on='key').drop('key', axis=1)

    # Merge to get dx_status
    phecode_occurrences['dx_status'] = 1
    w_big = all_combinations.merge(phecode_occurrences[['person_id', 'phecode', 'dx_status']],
                                   on=['person_id', 'phecode'], how='left')
    w_big['dx_status'].fillna(0, inplace=True)

    # Merge to get prevalence
    weights = w_big.merge(phecode_counts[['phecode', 'pred']], on='phecode', how='left')

    # Calculate weights
    weights['w'] = (1 - 2 * weights['dx_status']) * np.log10(
        weights['dx_status'] * weights['pred'] + (1 - weights['dx_status']) * (1 - weights['pred']))
    if not negative_weights:
        weights['w'] = weights['dx_status'] * weights['w']

    # Cleanup
    weights.drop('dx_status', axis=1, inplace=True)

    # Reorder columns
    weights = weights[['person_id', 'phecode', 'pred', 'w']]

    return weights


def get_weights_logistic(demos, phecode_occurrences, method_formula, negative_weights, n_jobs=1):
    """
    Calculate weights for each individual based on logistic regression of phecode occurrences.

    Parameters:
    - demos: pandas DataFrame containing demographic and other covariate information with 'person_id'.
    - phecode_occurrences: pandas DataFrame listing occurrences of phecodes for each 'person_id'.
    - method_formula: String representing the formula for logistic regression, e.g., "outcome ~ covariate1 + covariate2".
    - negative_weights: Boolean indicating whether to allow negative weights for individuals without occurrences of a phecode.
    - n_jobs: Number of parallel jobs for logistic regression calculations.

    Returns:
    A pandas DataFrame containing calculated weights for each combination of 'person_id' and 'phecode'.
    """

    def logistic_regression(group):
        person_ids = group['person_id'].unique()
        glm_input = demos[demos['person_id'].isin(person_ids)].copy()
        glm_input['dx_status'] = np.where(glm_input['person_id'].isin(person_ids), 1, 0)

        # Adjusting method formula to Python's statsmodels syntax could be complex and might require parsing
        # Assuming 'method_formula' is directly applicable or adjusted before being passed to this function
        formula = method_formula.replace("outcome", "dx_status")
        model = sm.GLM.from_formula(formula, data=glm_input, family=Binomial())
        result = model.fit()

        glm_input['pred'] = result.predict(glm_input)
        glm_input['w'] = (1 - 2 * glm_input['dx_status']) * np.log10(
            glm_input['dx_status'] * glm_input['pred'] + (1 - glm_input['dx_status']) * (1 - glm_input['pred']))

        if not negative_weights:
            glm_input['w'] = glm_input['dx_status'] * glm_input['w']

        return glm_input[['person_id', 'pred', 'w']]

    # Parallel processing (conceptual implementation)
    weights_dfs = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(logistic_regression, group) for _, group in phecode_occurrences.groupby('phecode')]
        for future in futures:
            weights_dfs.append(future.result())

    weights = pd.concat(weights_dfs, ignore_index=True)
    return weights





def get_weights_loglinear(demos, phecode_occurrences, method_formula, negative_weights, n_jobs=1):
    """
    Calculate weights for each individual based on linear regression of phecode occurrences.

    Parameters:
    - demos: pandas DataFrame containing demographic and other covariate information with 'person_id'.
    - phecode_occurrences: pandas DataFrame listing occurrences of phecodes for each 'person_id'.
    - method_formula: String representing the formula for linear regression, e.g., "outcome ~ covariate1 + covariate2".
    - negative_weights: Boolean indicating whether to allow negative weights for individuals without occurrences of a phecode.
    - n_jobs: Number of parallel jobs for linear regression calculations.

    Returns:
    A pandas DataFrame containing calculated weights for each combination of 'person_id' and 'phecode'.
    """

    def linear_regression(group):
        person_ids = group['person_id'].unique()
        lm_input = demos[demos['person_id'].isin(person_ids)].copy()
        lm_input = pd.merge(lm_input, group, on='person_id', how='left')

        lm_input['num_occurrences'].fillna(0, inplace=True)

        # Adjusting method formula for Python's statsmodels syntax
        formula = method_formula.replace("outcome", "np.log2(num_occurrences + 1)")
        result = smf.ols(formula, data=lm_input).fit()

        lm_input['pred'] = result.predict(lm_input)
        lm_input['w'] = np.log2(lm_input['num_occurrences'] + 1) - lm_input['pred']

        return lm_input[['person_id', 'phecode', 'pred', 'w']]

    # Parallel processing (conceptual implementation)
    weights_dfs = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(linear_regression, group) for _, group in phecode_occurrences.groupby('phecode')]
        for future in futures:
            weights_dfs.append(future.result())

    weights = pd.concat(weights_dfs, ignore_index=True)
    return weights



def get_weights_cox(demos, phecode_occurrences, method_formula, negative_weights):
    """
    Calculate weights using Cox Proportional Hazards model for phenotype risk scores.

    Parameters:
    - demos: DataFrame with demographics and 'first_age', 'last_age' for each 'person_id'.
    - phecode_occurrences: DataFrame with 'person_id', 'phecode', and 'occurrence_age'.
    - method_formula: Formula for the Cox model, not directly used as lifelines uses a DataFrame-based approach.
    - negative_weights: Whether to allow negative weights for non-occurrences.

    Returns:
    A DataFrame with calculated weights.
    """
    cox_fitter = CoxPHFitter()

    # Prepare the input DataFrame for Cox fitting
    cox_input = demos.copy()
    cox_input = cox_input.merge(phecode_occurrences, on='person_id', how='outer')

    # Initialize dx_status based on the presence of phecode and occurrence_age
    cox_input['dx_status'] = cox_input['phecode'].notna().astype(int)
    cox_input['age2'] = np.where(cox_input['dx_status'] == 1, cox_input['occurrence_age'], cox_input['last_age'])

    # Adjust ages to ensure they are within the observed span and not equal
    cox_input['age2'] = np.where(cox_input['age2'] == cox_input['first_age'], cox_input['age2'] + (1 / 365.25), cox_input['age2'])
    cox_input = cox_input[cox_input['age2'] > cox_input['first_age']]

    # Fit the Cox model - assuming 'time' and 'event' are specified correctly
    # Note: 'method_formula' is not used directly; users must ensure 'cox_input' DataFrame structure matches their formula
    cox_fitter.fit(cox_input[['age2', 'first_age', 'dx_status'] + [other covariates]], duration_col='age2', event_col='dx_status')

    # Predict the survival function for each individual
    cox_input['pred'] = cox_fitter.predict_partial_hazard(cox_input)

    # Calculate weights
    cox_input['w'] = (1 - 2 * cox_input['dx_status']) * np.log10(cox_input['dx_status'] * cox_input['pred'] + (1 - cox_input['dx_status']) * (1 - cox_input['pred']))

    if not negative_weights:
        cox_input['w'] = cox_input['dx_status'] * cox_input['w']

    return cox_input[['person_id', 'phecode', 'pred', 'w']]

# Note: This function assumes certain DataFrame structures and that 'first_age', 'last_age', etc., are correctly defined in 'demos'.
# The user must adapt the structure of 'demos' and 'phecode_occurrences' as needed.



def process_weighting_function(weights_func, demos, phecode_occurrences, method_formula, negative_weights, phe):
    """
    Wrapper function for calling the weighting function with the necessary parameters.
    This is needed to simplify the call within the parallel execution block.
    """
    # Subset phecode_occurrences for the specific phecode
    subset_occurrences = phecode_occurrences[phecode_occurrences['phecode'] == phe]

    # Call the appropriate weighting function
    return weights_func(demos, subset_occurrences, method_formula, negative_weights)


# Assume check_demos, check_phecode_occurrences, assert_flag, and
# check_method_formula are already defined elsewhere in your package.
# Also assume get_weights_prevalence, get_pre_calc_weights, get_weights_logistic,
# get_weights_loglinear, and get_weights_cox functions are defined as per the
# previous structure.

def get_weights(demos, phecode_occurrences, method='prevalence', method_formula=None,
                negative_weights=False, dopar=False):
    """
    Calculate phecode-specific weights for phenotype risk scores.

    Parameters:
        demos (DataFrame): Data containing one row per person in the cohort.
        phecode_occurrences (DataFrame): Data of phecode occurrences for each person in the cohort.
        method (str): Statistical model for calculating weights.
        method_formula (str or None): Formula for the model, required for logistic, loglinear, and cox methods.
        negative_weights (bool): Whether to allow negative weights for no occurrences.
        dopar (bool): Whether to run calculations in parallel.

    Returns:
        DataFrame: Weights calculated based on the specified method.
    """
    # Validate inputs
    check_demos(demos, method)
    check_phecode_occurrences(phecode_occurrences, demos, method)
    assert_flag(negative_weights)
    assert_flag(dopar)

    # Map method names to function calls
    method_functions = {
        'prevalence': get_weights_prevalence,
        'prevalence_precalc': get_pre_calc_weights,
        'logistic': get_weights_logistic,
        'loglinear': get_weights_loglinear,
        'cox': get_weights_cox
    }

    # Handle prevalence methods separately
    if method in ['prevalence', 'prevalence_precalc']:
        weights_func = method_functions[method]
        weights = weights_func(demos, phecode_occurrences, negative_weights)
        return weights

    # Validate method formula for other methods
    check_method_formula(method_formula, demos)

    # Get the unique phecodes
    phecodes = phecode_occurrences['phecode'].unique()

    weights = []

    # Prepare parallel execution
    if dopar:
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor() as executor:
            # Prepare future tasks
            futures = {
                executor.submit(process_weighting_function, method_functions[method], demos, phecode_occurrences,
                                method_formula, negative_weights, phe): phe for phe in phecodes
            }
            # Collect results as they complete
            for future in as_completed(futures):
                phe = futures[future]
                try:
                    weight_result = future.result()
                    weights.append(weight_result)
                except Exception as exc:
                    print(f'Phecode {phe} generated an exception: {exc}')
    else:
        # Sequential execution
        for phe in phecodes:
            weight_result = process_weighting_function(method_functions[method], demos, phecode_occurrences,
                                                       method_formula, negative_weights, phe)
            weights.append(weight_result)

        # Combine all weights into a single DataFrame
    weights_df = pd.concat(weights)

    return weights_df














