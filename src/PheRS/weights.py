import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import warnings

import polars as pl
import numpy as np
from numpy import log2
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Binomial
from lifelines import CoxPHFitter

from patsy import EvalEnvironment
# noinspection PyUnresolvedReferences,PyProtectedMember
from pheRS.src.PheRS import utils


class Weights:
    """
    A class to calculate weights for phenotype risk scores.
    """

    def __init__(self, platform="aou", demo_df_path=None, suppress_warnings=True):
        self.platform = platform
        self.suppress_warnings = suppress_warnings
        self.n_threads = round(os.cpu_count() * 2 / 3)

        if platform == "aou":
            self.demos = utils.get_allofus_demo()
        elif platform == "custom":
            if demo_df_path is not None:
                print("\033[1mLoading user's demographic data from file...")
                self.demos = pl.read_csv(demo_df_path)
            else:
                print("demo_df_path is required for custom platform.")
                sys.exit(0)
        else:
            print('Invalid platform. Parameter platform only accepts "aou" (All of Us) or "custom".')
            sys.exit(0)

        print("\033[1mDone!")

    def get_weights_prevalence(self, phecode_occurrences_path=None, negative_weights=False, n_jobs=1,
                               output_file_name=None):
        start_time = datetime.now()
        print(f"Prevalence weights calculation started at: {start_time}")

        if phecode_occurrences_path is None:
            print("phecode_occurrences path is required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos.clone()

        # Calculate prevalence (pred) for each phecode
        phecode_counts = phecode_occurrences.group_by('phecode').agg(pl.count('person_id')).rename(
            {'person_id': 'count'})
        phecode_counts = phecode_counts.with_columns((phecode_counts['count'] / len(demos)).alias('pred'))

        # Prepare a dataframe with all combinations of person_id and phecode
        all_combinations = demos.select(['person_id']).join(phecode_counts, how='cross')

        # Function to process each phecode
        def weight_prevalence(phe):
            phe_occurrences = phecode_occurrences.filter(pl.col('phecode') == phe).with_columns(
                pl.lit(1).alias('dx_status'))
            w_big = all_combinations.join(phe_occurrences, on=['person_id', 'phecode'], how='left').fill_null(0)
            weights = w_big.join(phecode_counts.filter(pl.col('phecode') == phe).select(['phecode', 'pred']),
                                 on='phecode', how='left')
            weights = weights.with_columns(((1 - 2 * weights['dx_status']) * np.log10(
                weights['dx_status'] * weights['pred'] + (1 - weights['dx_status']) * (1 - weights['pred']))).alias(
                'w'))
            if not negative_weights:
                weights = weights.with_columns((weights['dx_status'] * weights['w']).alias('w'))
                weights = weights.with_columns(pl.when(pl.col('w') == -0.0).then(0.0).otherwise(pl.col('w')).alias('w'))
            return weights.select(['person_id', 'phecode', 'pred', 'w']).unique()

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(weight_prevalence, phe) for phe in unique_phecodes]
            for future in tqdm(as_completed(futures), total=len(unique_phecodes), desc="Prevalence Calculation"):
                if self.suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        weight_result = future.result()
                else:
                    weight_result = future.result()

                if weight_result is not None:
                    weights_dfs.append(weight_result)

        weights = pl.concat(weights_dfs)
        weights = weights.unique(subset=['person_id', 'phecode'])
        weights = weights.with_columns(weights["phecode"].cast(pl.Utf8))

        # Report result
        utils.report_result(weights, placeholder='weights_prevalence', output_file_name=output_file_name)

        end_time = datetime.now()
        print(f"Prevalence weights calculation completed at: {end_time}")
        duration = end_time - start_time
        print(f"Total duration: {duration}")

    def get_weights_logistic(self, phecode_occurrences_path=None, method_formula=None, negative_weights=False,
                             n_jobs=1, output_file_name=None):
        start_time = datetime.now()
        print(f"Logistic weights calculation started at: {start_time}")

        if phecode_occurrences_path is None and method_formula is None:
            print("Both phecode_occurrences path and method_formula are required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos.clone()

        def logistic_regression(phe):
            phe_sub = phecode_occurrences.filter(pl.col('phecode') == phe).select(['person_id']).unique()
            glm_input = demos.join(phe_sub.with_columns(pl.lit(phe).alias('phecode')), on='person_id', how='left')
            glm_input = glm_input.with_columns((pl.col('phecode').is_not_null().cast(pl.Int64)).alias('dx_status'))
            formula = f"dx_status ~ {method_formula}"
            model = glm(formula, data=glm_input.to_pandas(), family=Binomial()).fit()

            glm_input = glm_input.with_columns([pl.Series(model.predict(glm_input.to_pandas())).alias('pred')])
            glm_input = glm_input.with_columns(((1 - 2 * glm_input['dx_status']) * np.log10(
                glm_input['dx_status'] * glm_input['pred'] + (1 - glm_input['dx_status']) * (
                        1 - glm_input['pred']))).alias('w'))

            if not negative_weights:
                glm_input = glm_input.with_columns((glm_input['dx_status'] * glm_input['w']).alias('w'))
                glm_input = glm_input.with_columns(
                    pl.when(pl.col('w') == -0.0).then(0.0).otherwise(pl.col('w')).alias('w'))

            return glm_input.select(['person_id', 'phecode', 'pred', 'w']).unique()

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(logistic_regression, phe) for phe in unique_phecodes]
            for future in tqdm(as_completed(futures), total=len(unique_phecodes), desc="Logistic Regression"):
                if self.suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        weight_result = future.result()
                else:
                    weight_result = future.result()

                if weight_result is not None:
                    weights_dfs.append(weight_result)

        weights = pl.concat(weights_dfs)
        weights = weights.unique(subset=['person_id', 'phecode'])
        weights = weights.with_columns(weights["phecode"].cast(pl.Utf8))

        # Report result
        utils.report_result(weights, placeholder='weights_logistic', output_file_name=output_file_name)

        end_time = datetime.now()
        print(f"Logistic weights calculation completed at: {end_time}")
        duration = end_time - start_time
        print(f"Total duration: {duration}")

    def get_weights_loglinear(self, phecode_occurrences_path=None, method_formula=None, negative_weights=False,
                              n_jobs=1, output_file_name=None):
        start_time = datetime.now()
        print(f"Loglinear weights calculation started at: {start_time}")

        if phecode_occurrences_path is None and method_formula is None:
            print("Both phecode_occurrences path and method_formula are required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos.clone()

        def linear_regression(phe):
            phe_sub = phecode_occurrences.filter(pl.col('phecode') == phe).select(
                ['person_id', 'num_occurrences']).unique()
            lm_input = demos.join(phe_sub, on='person_id', how='left').fill_null(0)
            lm_input = lm_input.with_columns((pl.lit(phe)).alias('phecode'))
            formula = f"log2(num_occurrences + 1) ~ {method_formula}"

            eval_env = EvalEnvironment.capture(0)
            eval_env.namespace['log2'] = log2
            model = ols(formula, data=lm_input.to_pandas(), eval_env=eval_env).fit()

            lm_input = lm_input.with_columns([pl.Series(model.predict(lm_input.to_pandas())).alias('pred')])
            lm_input = lm_input.with_columns((np.log2(lm_input['num_occurrences'] + 1) - lm_input['pred']).alias('w'))

            if not negative_weights:
                lm_input = lm_input.with_columns(
                    pl.when(pl.col('num_occurrences') == 0).then(0).otherwise(lm_input['w']).alias('w'))

            return lm_input.select(['person_id', 'phecode', 'pred', 'w']).unique()

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(linear_regression, phe) for phe in unique_phecodes]
            for future in tqdm(as_completed(futures), total=len(unique_phecodes), desc="Linear Regression"):
                if self.suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        weight_result = future.result()
                else:
                    weight_result = future.result()

                if weight_result is not None:
                    weights_dfs.append(weight_result)

        weights = pl.concat(weights_dfs)
        weights = weights.unique(subset=['person_id', 'phecode'])
        weights = weights.with_columns(weights["phecode"].cast(pl.Utf8))

        # Report result
        utils.report_result(weights, placeholder='weights_linear', output_file_name=output_file_name)

        end_time = datetime.now()
        print(f"Loglinear weights calculation completed at: {end_time}")
        duration = end_time - start_time
        print(f"Total duration: {duration}")

    def get_weights_cox(self, phecode_occurrences_path=None, method_formula=None, negative_weights=False, n_jobs=1,
                        output_file_name=None):
        start_time = datetime.now()
        print(f"Cox weights calculation started at: {start_time}")

        if phecode_occurrences_path is None:
            print("phecode_occurrences path is required to calculate weights.")
            sys.exit(0)

        if method_formula is not None and not isinstance(method_formula, list):
            print("method_formula must be provided as a list of column names or be None.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos.clone()

        def cox_regression(phe):
            phe_sub = phecode_occurrences.filter(pl.col('phecode') == phe).select(
                ['person_id', 'occurrence_age']).unique()
            cox_input = demos.join(phe_sub, on='person_id', how='left').with_columns(
                (pl.col('occurrence_age').is_not_null().cast(pl.Int64)).alias('dx_status'))
            cox_input = cox_input.with_columns(pl.lit(phe).alias('phecode'))
            cox_input = cox_input.with_columns(pl.when(pl.col('dx_status') == 1).then(pl.col('occurrence_age'))
                                               .otherwise(pl.col('last_age')).alias('age2'))
            cox_input = cox_input.with_columns(pl.when(pl.col("age2") == pl.col("first_age")).
                                               then(pl.col("age2") + (1 / 365.25)).otherwise(pl.col("age2"))
                                               .alias("age2"))
            cox_input = cox_input.filter(pl.col('age2') > pl.col('first_age'))

            cox_fitter = CoxPHFitter()

            if method_formula is None:
                cox_fitter.fit(cox_input.select(['age2', 'first_age', 'dx_status']).to_pandas(),
                               duration_col='age2', event_col='dx_status')
            else:
                cox_fitter.fit(cox_input.select(['age2', 'first_age', 'dx_status'] + method_formula).to_pandas(),
                               duration_col='age2', event_col='dx_status')

            cox_input = cox_input.with_columns([pl.Series(
                (1 - np.exp(-cox_fitter.predict_partial_hazard(cox_input.to_pandas())))).alias('pred')])
            cox_input = cox_input.with_columns(((1 - 2 * cox_input['dx_status']) * np.log10(
                cox_input['dx_status'] * cox_input['pred'] + (1 - cox_input['dx_status']) * (
                        1 - cox_input['pred']))).alias('w'))

            if not negative_weights:
                cox_input = cox_input.with_columns((cox_input['dx_status'] * cox_input['w']).alias('w'))
                cox_input = cox_input.with_columns(
                    pl.when(pl.col('w') == -0.0).then(0.0).otherwise(pl.col('w')).alias('w'))

            return cox_input.select(['person_id', 'phecode', 'pred', 'w']).unique()

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(cox_regression, phe) for phe in unique_phecodes]
            for future in tqdm(as_completed(futures), total=len(unique_phecodes), desc="Cox Regression"):
                if self.suppress_warnings:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        weight_result = future.result()
                else:
                    weight_result = future.result()

                if weight_result is not None:
                    weights_dfs.append(weight_result)

        weights = pl.concat(weights_dfs)
        weights = weights.unique(subset=['person_id', 'phecode'])
        weights = weights.with_columns(weights["phecode"].cast(pl.Utf8))

        # Report result
        utils.report_result(weights, placeholder='weights_cox', output_file_name=output_file_name)

        end_time = datetime.now()
        print(f"Cox weights calculation completed at: {end_time}")
        duration = end_time - start_time
        print(f"Total duration: {duration}")

    def get_weights(self, phecode_occurrences_path=None, method='prevalence', method_formula=None,
                    negative_weights=False, n_jobs=1, output_file_name=None):
        if phecode_occurrences_path is None:
            print("phecode_occurrences path is required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        phecode_occurrences = phecode_occurrences.with_columns(phecode_occurrences['phecode'].cast(pl.Utf8))
        demos = self.demos.clone()

        utils.check_demos(demos, method)
        utils.check_phecode_occurrences(phecode_occurrences, demos, method)
        if not isinstance(negative_weights, bool):
            print('"negative_weights" must be True or False.')
            sys.exit(0)

        if method == 'prevalence':
            self.get_weights_prevalence(phecode_occurrences_path, negative_weights, n_jobs,
                                        output_file_name=output_file_name)

        if method == 'logistic':
            if method_formula is None:
                print("method_formula cannot be \"None\". Required to implement \"check_method_formula\" function .")
                sys.exit(0)
            utils.check_method_formula(method_formula, demos)
            self.get_weights_logistic(phecode_occurrences_path, method_formula, negative_weights, n_jobs,
                                      output_file_name=output_file_name)

        elif method == 'loglinear':
            if method_formula is None:
                print("method_formula cannot be \"None\". Required to implement \"check_method_formula\" function .")
                sys.exit(0)
            utils.check_method_formula(method_formula, demos)
            self.get_weights_loglinear(phecode_occurrences_path, method_formula, negative_weights, n_jobs,
                                       output_file_name=output_file_name)

        elif method == 'cox':
            self.get_weights_cox(phecode_occurrences_path, method_formula, negative_weights, n_jobs,
                                 output_file_name=output_file_name)
