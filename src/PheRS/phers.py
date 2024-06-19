import polars as pl
import sys
import statsmodels.formula.api as smf
# noinspection PyUnresolvedReferences,PyProtectedMember
from PheRS import queries, utils


def get_scores(weights_path=None, disease_phecode_map=None, disease_id=None, output_file_name=None):
    """
    Aggregates phecode-specific weights into overall disease risk scores for individuals.

    Parameters:
    - weights: polars.DataFrame containing phecode weights for each person.
        Must have columns 'person_id', 'phecode', and 'w'.
    - disease_phecode_map: polars.DataFrame containing mapping from phecodes to diseases.
        Must have columns 'phecode' and 'disease_id'.

    Returns:
    - pandas.DataFrame with columns 'person_id', 'disease_id', and 'score', representing the aggregated
        risk score of diseases per person.
    """

    # noinspection PyGlobalUndefined
    if weights_path is not None and disease_phecode_map is not None:
        weights = pl.read_csv(weights_path)
        weights = weights.with_columns(weights["phecode"].cast(pl.Utf8))
        disease_phecode_map = pl.read_csv(disease_phecode_map)
        disease_phecode_map = disease_phecode_map.with_columns(disease_phecode_map["phecode"].cast(pl.Utf8))
        if "disease_id" not in disease_phecode_map.columns and "phecode" not in disease_phecode_map.columns:
            print("Disease_phecode_map file must contain \"disease_id\" and \"phecode\" columns!")
            sys.exit(0)
    else:
        print("Both \"weight_path\" and \"disease_phecode_map\" dataframe are required."
              "Please provide a valid file path.")
        sys.exit(0)

    # Validation checks
    utils.check_weights(weights)
    utils.check_disease_phecode_map(disease_phecode_map)

    if disease_id is not None:
        disease_phecode_map = disease_phecode_map.filter(pl.col("disease_id") == disease_id)

    # Merge weights with disease_phecode_map
    merged_data = weights.join(disease_phecode_map, on='phecode', how='inner')

    # Calculate scores by summing weights per person per disease
    scores = merged_data.group_by(['person_id', 'disease_id']).agg(
        pl.col('w').sum().alias('score')
    )

    # report result
    utils.report_result(scores, placeholder="disease_score",
                        output_file_name=output_file_name)


def get_residual_scores(platform='aou', demos_path=None, scores_path=None, lm_formula=None,
                        output_file_name=None):
    """
    Calculates residual scores for diseases by adjusting for demographics using a linear model.

    Parameters:
    - demo: polars.DataFrame containing demographic data for each person. Must have a column 'person_id'.
    - scores: polars.DataFrame containing disease scores for each person. Must have columns 'person_id',
        'disease_id', and 'score'.
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
    # utils.check_lm_formula(lm_formula, demo)

    # Merge scores with demographic data using Polars
    r_input = scores.join(demo, on='person_id', how='inner')

    # Convert Polars DataFrame to pandas DataFrame for statsmodels
    r_input_pd = r_input.to_pandas()

    # Fit linear model and calculate residuals using pandas and statsmodels
    lm_model = smf.ols(formula=lm_formula, data=r_input_pd).fit()
    r_input_pd['resid_score'] = lm_model.get_influence().resid_studentized_internal

    # Convert results back to Polars DataFrame
    r_scores = pl.from_pandas(r_input_pd[['person_id', 'disease_id', 'score', 'resid_score']])

    # report result
    utils.report_result(r_scores, placeholder="residual_score",
                        output_file_name=output_file_name)


if __name__ == "__main__":
#    get_scores(weights_path="final_weights_linear.csv", disease_phecode_map="disease_phecode.csv",
#               disease_id=None, output_file_name=None)

#    get_residual_scores(platform='custom', demos_path='/Users/dayoshittu/Downloads/demo_data_sample.csv',
#                        scores_path='disease_score.csv', lm_formula='score ~ sex + last_age', output_file_name=None)
