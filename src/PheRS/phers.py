import pandas as pd
import os
import polars as pl
import sys
# import statsmodels.api as sm
import statsmodels.formula.api as smf
# noinspection PyUnresolvedReferences,PyProtectedMember
from PheRS import queries, utils


class Phers:
    """
        Class phers implements three different functions
         - get_phecode_occurrences that maps icd_occurrences with the icd_phecode_map
         - get_scores
         - get_residual_scores
        Currently, supports ICD code extraction for All of Us OMOP data.
        For other databases, user is expected to provide an ICD code table for all participants in cohort of interest.
    """

    def __init__(self, platform="aou", icd_df_path=None):
        """
               Instantiate based on parameter db
               :param platform: supports:
                   "aou": All of Us OMOP database
                   "custom": other databases; icd_df must be not None if db = "custom"
               :param icd_df_path: path to ICD table csv file; required columns are "person_id", "ICD", and
               "flag" ("vocabulary_id");
                   "flag"("vocabulary_id") values should be "ICD9CM" or "ICD10CM"
               """
        self.weights = None  # Read in the CSV file
        self.platform = platform

        if platform == "aou":
            self.cdr = os.getenv("WORKSPACE_CDR")
            self.icd_query = queries.all_icd_query(self.cdr)
            print("\033[1mStart querying ICD codes...")
            self.icd_events = utils.polars_gbq(self.icd_query)

        elif platform == "custom":
            if icd_df_path is not None:
                print("\033[1mLoading user's ICD data from file...")
                self.icd_events = pl.read_csv(icd_df_path,
                                              dtypes={"ICD": str})
            else:
                print("icd_df_path is required for custom platform.")
                sys.exit(0)
        else:
            print("Invalid platform. Parameter platform only accepts \"aou\" (All of Us) or \"custom\".")
            sys.exit(0)

        # add flag column if not exist
        if "flag" not in self.icd_events.columns:
            self.icd_events = self.icd_events.with_columns(
                pl.when((pl.col("vocabulary_id") == "ICD9") |
                        (pl.col("vocabulary_id") == "ICD9CM"))
                .then(9)
                .when((pl.col("vocabulary_id") == "ICD10") |
                      (pl.col("vocabulary_id") == "ICD10CM"))
                .then(10)
                .otherwise(0)
                .alias("flag")
                .cast(pl.Int8)
            )
        else:
            self.icd_events = self.icd_events.with_columns(pl.col("flag").cast(pl.Int8))

        print("\033[1mDone!")

    def get_phecode_occurrences(self, icd_phecode_map_path=None, dx_icd=None,
                                phecode_version="X", icd_version="US", output_file_name=None):
        """
        Maps ICD occurrences to phecodes by merging ICD occurrence data with a mapping table.

        Parameters:
        - icd_occurrences: pandas.DataFrame containing ICD code occurrences for each person in the cohort.
            Must have columns 'icd', 'flag', and 'person_id'.
        - icd_phecode_map: pandas.DataFrame containing the mapping from ICD codes to phecodes.
            Must have columns 'icd', 'flag', and 'phecode'.
        - dx_icd: Optional. pandas.DataFrame containing ICD codes to be removed. Must have columns 'icd' and 'flag'.
            If None, no ICD codes are removed.
        - phecode_version: Version of phecode to be merged. Either X or 1.2
        - icd_version: Version of ICD code to be merged. Either US, WHO or custom

        Returns:
        - A pandas.DataFrame with columns 'person_id' and 'phecode', representing the occurrence of phecodes per person.
        """

        # load phecode mapping file by version or by custom path
        # noinspection PyGlobalUndefined
        global icd_occurrences
        icd_phecode_map = utils.get_phecode_mapping(
            phecode_version=phecode_version,
            icd_version=icd_version,
            icd_phecode_map_path=icd_phecode_map_path,
            keep_all_columns=False
        )

        # make a copy of self.icd_events
        icd_events = self.icd_events.clone()

        # Required columns
        icd_events = icd_events[["person_id", "ICD", "flag"]]

        print()
        print(f"\033[1mMapping ICD codes to phecode {phecode_version}...")

        # Validation checks
        utils.check_icd_occurrences(icd_events)
        utils.check_icd_phecode_map(icd_phecode_map)

        # Remove ICD codes if dx_icd is provided
        if dx_icd is not None and not dx_icd.empty:
            dx_icd = utils.to_polars(dx_icd)
            dx_icd = dx_icd.with_columns(pl.lit(1).alias('merge_indicator'))  # Temporary merge indicator

            icd_occurrences = icd_events.join(dx_icd, on=['ICD', 'flag'], how='left')
            icd_occurrences = icd_occurrences.filter(pl.col("merge_indicator").is_null()).drop("merge_indicator")

        # Merge icd occurrences with icd phecode map
        if phecode_version == "X":
            phecode_occurrences = icd_occurrences.join(icd_phecode_map[['ICD', 'flag', 'phecode']], on=['ICD', 'flag'],
                                                       how='inner')
        elif phecode_version == "1.2":
            phecode_occurrences = icd_occurrences.join(icd_phecode_map[['ICD', 'flag', 'phecode']], on=['ICD', 'flag'],
                                                       how='inner')
            phecode_occurrences = phecode_occurrences.rename({"phecode_unrolled": "phecode"})
        else:
            phecode_occurrences = pl.DataFrame()
        phecode_occurrences = phecode_occurrences.select(['person_id', 'phecode']).unique()

        if not phecode_occurrences.is_empty():
            phecode_occurrences = phecode_occurrences.group_by(["person_id", "phecode"]).len().rename({"len": "count"})

        # report result
        utils.report_result(phecode_occurrences, phecode_version=None, placeholder=None,
                            output_file_name=output_file_name)

    def get_scores(self, weight_path=None, disease_phecode_map_path=None, output_file_name=None):
        """
        Aggregates phecode-specific weights into overall disease risk scores for individuals.

        Parameters:
        - weights: pandas.DataFrame containing phecode weights for each person.
            Must have columns 'person_id', 'phecode', and 'w'.
        - disease_phecode_map: pandas.DataFrame containing mapping from phecodes to diseases.
            Must have columns 'phecode' and 'disease_id'.

        Returns:
        - pandas.DataFrame with columns 'person_id', 'disease_id', and 'score', representing the aggregated
            risk score of diseases per person.
        """

        # noinspection PyGlobalUndefined
        global weights, disease_phecode_map
        if weight_path is not None or disease_phecode_map_path is not None:
            if weight_path is not None:
                weights = pl.read_csv(weight_path)
            elif disease_phecode_map_path is not None:
                disease_phecode_map = pl.read_csv(disease_phecode_map_path)
                if "disease_id" not in disease_phecode_map.columns and "phecode" not in disease_phecode_map.columns:
                    print("Disease_phecode_map file must contain \"disease_id\" and \"phecode\" columns!")
                    sys.exit(0)
            else:
                print("Either \"weight_path\" or \"disease_phecode_map\" dataframe is required."
                      "Please provide a valid file path.")
                sys.exit(0)
        else:
            print("Both \"weight_path\" and \"disease_phecode_map\" dataframe are required."
                  "Please provide a valid file path.")
            sys.exit(0)

        # Validation checks (simplified for this example)
        utils.check_weights(weights)
        utils.check_disease_phecode_map(disease_phecode_map)

        # Merge weights with disease phecode map
        merged_data = weights.join(disease_phecode_map, on='phecode', how='inner')

        # Calculate scores by summing weights per person per disease
        scores = merged_data.groupby(['person_id', 'disease_id']).agg(
            pl.col('w').sum().alias('score')
        )

        # report result
        utils.report_result(scores, phecode_version=None, placeholder="disease_score",
                            output_file_name=output_file_name)

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
