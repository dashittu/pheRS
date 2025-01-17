import polars as pl
import sys
import statsmodels.formula.api as smf
# noinspection PyUnresolvedReferences,PyProtectedMember
from pheRS.src.PheRS import queries, utils
    
def get_scores(weights_path=None, output_file_name=None):
    """
    Aggregates phecode-specific weights into overall disease risk scores for individuals.

    Parameters:
    - weights: polars.DataFrame containing phecode weights for each person.
        Must have columns 'person_id', 'phecode', and 'w'.
    - output_file_name: Optional path to save the final Phenotype Risk Score (PheRS)
    
    Returns:
    - pandas.DataFrame with columns 'person_id' and 'score', representing the aggregated
         risk score for each individual (sum of all phecode weights).
    """

    # noinspection PyGlobalUndefined
    if weights_path is None:
        print("weights_path is required. Please provide a valid file path.")
        sys.exit(0)
        
    weights = pl.read_csv(weights_path)
    weights = weights.with_columns(weights["phecode"].cast(pl.Utf8))

    # Validation checks
    utils.check_weights(weights)

    # Calculate scores by summing all weights per person
    scores = weights.group_by(['person_id']).agg(
        pl.col("w").sum().alias("score")
    )

    # report result
    utils.report_result(scores, placeholder="phenotype_risk_score",
                        output_file_name=output_file_name)


def get_residual_scores(platform='aou', demos_path=None, scores_path=None, lm_formula=None,
                        output_file_name=None):
    """
    Calculates residual scores for diseases by adjusting for demographics using a linear model.

    Parameters:
    - demo: polars.DataFrame containing demographic data for each person. Must have a column 'person_id'.
    - scores: polars.DataFrame containing disease scores for each person. Must have columns 'person_id'
        and 'score'.
    - lm_formula: string representing the formula for the linear model, e.g., 'score ~ age + sex'.

    Returns:
    - polars.DataFrame with columns 'person_id', 'disease_id', 'score', and 'resid_score', where 'resid_score' is
        the standardized residual from the linear model.
    """

    if platform == "aou":
        demo = utils.get_allofus_demo()
    elif platform == "custom":
        if demos_path is not None:
            print("\033[1mLoading user's demography data from file...")
            demo = pl.read_csv(demos_path)
        else:
            print("demos_path is required for custom platform.")
            sys.exit(0)
    else:
        print("Invalid platform. Parameter platform only accepts \"aou\" (All of Us) or \"custom\".")
        sys.exit(0)

    print("\033[1mDone!")

    if scores_path is not None:
        scores = pl.read_csv(scores_path)
    else:
        print('"scores_path" dataframe is required. Please provide a valid file path.')
        sys.exit(0)

    # Validation checks
    utils.check_demos(demo)
    utils.check_scores(scores)
                        
    if not lm_formula:
        print("No linear model formula provided. Please supply lm_formula, e.g., 'score ~ age + sex'.")
        sys.exit(0)

    # Merge scores with demographic data using Polars
    r_input = scores.join(demo, on='person_id', how='inner')

    # Convert Polars DataFrame to pandas DataFrame for statsmodels
    r_input_pd = r_input.to_pandas()

    # Fit linear model and calculate residuals using pandas and statsmodels
    lm_model = smf.ols(formula=lm_formula, data=r_input_pd).fit()
    r_input_pd['resid_score'] = lm_model.get_influence().resid_studentized_internal

    # Convert results back to Polars DataFrame
    r_scores = pl.from_pandas(r_input_pd[['person_id', 'score', 'resid_score']])

    # report result
    utils.report_result(r_scores, placeholder="residual_score",
                        output_file_name=output_file_name)

    print("\033[1mResidual scores calculation complete!\n")
