import polars as pl
import numpy as np
import sys
from statsmodels.formula.api import glm, ols
from statsmodels.genmod.families import Binomial
from lifelines import CoxPHFitter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from src.PheRS import utils


def process_weighting_function(weights_func, phecode_occurrences_path, method_formula, negative_weights, phe):
    phecode_occurrences = pl.read_csv(phecode_occurrences_path)
    return weights_func(phecode_occurrences, method_formula, negative_weights, phe)


class Weights:
    """
    A class to calculate weights for phenotype risk scores.
    """

    def __init__(self, platform="aou", demo_df_path=None):
        self.platform = platform
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

    def get_weights_prevalence(self, phecode_occurrences_path=None, negative_weights=False):
        if phecode_occurrences_path is None:
            print("phecode_occurrences path is required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos

        # Calculate prevalence (pred) for each phecode
        phecode_counts = phecode_occurrences.groupby('phecode').agg(pl.count('person_id')).rename(
            {'person_id': 'count'})
        phecode_counts = phecode_counts.with_column((phecode_counts['count'] / len(demos)).alias('pred'))

        # Prepare a dataframe with all combinations of person_id and phecode
        all_combinations = demos.select(['person_id']).join(phecode_counts, how='cross')

        # Merge to get dx_status
        phecode_occurrences = phecode_occurrences.with_column(pl.lit(1).alias('dx_status'))
        w_big = all_combinations.join(phecode_occurrences, on=['person_id', 'phecode'], how='left').fillna(0)

        # Merge to get prevalence
        weights = w_big.join(phecode_counts.select(['phecode', 'pred']), on='phecode', how='left')

        # Calculate weights
        weights = weights.with_column(((1 - 2 * weights['dx_status']) * np.log10(
            weights['dx_status'] * weights['pred'] + (1 - weights['dx_status']) * (1 - weights['pred']))).alias('w'))
        if not negative_weights:
            weights = weights.with_column((weights['dx_status'] * weights['w']).alias('w'))

        # Reorder columns
        weights = weights.select(['person_id', 'phecode', 'pred', 'w'])

        return weights

    def get_weights_logistic(self, phecode_occurrences_path=None, method_formula=None, negative_weights=False,
                             n_jobs=1):
        if phecode_occurrences_path is None or method_formula is None:
            print("Both phecode_occurrences path and method_formula are required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos

        def logistic_regression(phe):
            phe_sub = phecode_occurrences.filter(pl.col('phecode') == phe).select(['person_id']).unique()
            glm_input = demos.join(phe_sub.with_column(pl.lit(phe).alias('phecode')), on='person_id', how='left')
            glm_input = glm_input.with_column((pl.col('phecode').is_not_null().cast(pl.Int64)).alias('dx_status'))
            formula = f"dx_status ~ {method_formula}"
            model = glm(formula, data=glm_input.to_pandas(), family=Binomial()).fit()

            glm_input = glm_input.with_column((model.predict(glm_input.to_pandas())).alias('pred'))
            glm_input = glm_input.with_column(((1 - 2 * glm_input['dx_status']) * np.log10(
                glm_input['dx_status'] * glm_input['pred'] + (1 - glm_input['dx_status']) * (
                        1 - glm_input['pred']))).alias('w'))

            if not negative_weights:
                glm_input = glm_input.with_column((glm_input['dx_status'] * glm_input['w']).alias('w'))

            return glm_input.select(['person_id', 'phecode', 'pred', 'w'])

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(logistic_regression, phe) for phe in unique_phecodes]
            for future in futures:
                weights_dfs.append(future.result())

        weights = pl.concat(weights_dfs)
        return weights

    def get_weights_loglinear(self, phecode_occurrences_path=None, method_formula=None, negative_weights=False,
                              n_jobs=1):
        if phecode_occurrences_path is None or method_formula is None:
            print("Both phecode_occurrences path and method_formula are required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos

        def linear_regression(phe):
            phe_sub = phecode_occurrences.filter(pl.col('phecode') == phe).select(
                ['person_id', 'num_occurrences']).unique()
            lm_input = demos.join(phe_sub, on='person_id', how='left').fillna(0)
            lm_input = lm_input.with_column((pl.lit(phe)).alias('phecode'))
            formula = f"log2(num_occurrences + 1) ~ {method_formula}"
            model = ols(formula, data=lm_input.to_pandas()).fit()

            lm_input = lm_input.with_column((model.predict(lm_input.to_pandas())).alias('pred'))
            lm_input = lm_input.with_column((np.log2(lm_input['num_occurrences'] + 1) - lm_input['pred']).alias('w'))

            if not negative_weights:
                lm_input = lm_input.with_column(
                    pl.when(pl.col('num_occurrences') == 0).then(0).otherwise(lm_input['w']).alias('w'))

            return lm_input.select(['person_id', 'phecode', 'pred', 'w'])

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(linear_regression, phe) for phe in unique_phecodes]
            for future in futures:
                weights_dfs.append(future.result())

        weights = pl.concat(weights_dfs)
        return weights

    def get_weights_cox(self, phecode_occurrences_path=None, covariates=None, negative_weights=False, n_jobs=1):
        if phecode_occurrences_path is None:
            print("phecode_occurrences path is required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos

        def cox_regression(phe):
            phe_sub = phecode_occurrences.filter(pl.col('phecode') == phe).select(
                ['person_id', 'occurrence_age']).unique()
            cox_input = demos.join(phe_sub, on='person_id', how='left').with_column(
                (pl.col('phecode').is_not_null().cast(pl.Int64)).alias('dx_status'))
            cox_input = cox_input.with_column(pl.when(pl.col('dx_status') == 1).then(pl.col('occurrence_age'))
                                               .otherwise(pl.col('last_age')).alias('age2'))
            cox_input = cox_input.with_columns(pl.when(pl.col("age2") == pl.col("first_age")).
                                               then(pl.col("age2") + (1 / 365.25)).otherwise(pl.col("age2"))
                                               .alias("age2"))
            cox_input = cox_input.filter(pl.col('age2') > pl.col('first_age'))

            cox_fitter = CoxPHFitter()
            cox_fitter.fit(cox_input[['age2', 'first_age', 'dx_status'] + covariates],
                           duration_col='age2', event_col='dx_status')

            cox_input = cox_input.with_column((1 - np.exp(-cox_fitter.predict_partial_hazard(cox_input))).alias('pred'))
            cox_input = cox_input.with_column(((1 - 2 * cox_input['dx_status']) * np.log10(
                cox_input['dx_status'] * cox_input['pred'] + (1 - cox_input['dx_status']) * (
                        1 - cox_input['pred']))).alias('w'))

            if not negative_weights:
                cox_input = cox_input.with_column((cox_input['dx_status'] * cox_input['w']).alias('w'))

            return cox_input.select(['person_id', 'phecode', 'pred', 'w'])

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights_dfs = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(cox_regression, phe) for phe in unique_phecodes]
            for future in futures:
                weights_dfs.append(future.result())

        weights = pl.concat(weights_dfs)
        return weights

    def get_weights(self, phecode_occurrences_path=None, method='prevalence', method_formula=None,
                    negative_weights=False, dopar=False):
        if phecode_occurrences_path is None:
            print("phecode_occurrences path is required to calculate weights.")
            sys.exit(0)

        phecode_occurrences = pl.read_csv(phecode_occurrences_path)
        demos = self.demos

        utils.check_demos(demos, method)
        utils.check_phecode_occurrences(phecode_occurrences, demos, method)
        if not isinstance(negative_weights, bool):
            print('"negative_weights" must be True or False.')
            sys.exit(0)
        if not isinstance(dopar, bool):
            print('"dopar" must be True or False.')
            sys.exit(0)

        method_functions = {
            'prevalence': self.get_weights_prevalence,
            'logistic': self.get_weights_logistic,
            'loglinear': self.get_weights_loglinear,
            'cox': self.get_weights_cox
        }

        if method == 'prevalence':
            return self.get_weights_prevalence(phecode_occurrences_path, negative_weights)

        utils.check_method_formula(method_formula, demos)

        unique_phecodes = phecode_occurrences['phecode'].unique()
        weights = []

        if dopar:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(process_weighting_function, method_functions[method], phecode_occurrences_path,
                                    method_formula, negative_weights, phe): phe for phe in unique_phecodes}
                for future in as_completed(futures):
                    phe = futures[future]
                    try:
                        weight_result = future.result()
                        weights.append(weight_result)
                    except Exception as exc:
                        print(f'Phecode {phe} generated an exception: {exc}')
        else:
            for phe in unique_phecodes:
                weight_result = process_weighting_function(method_functions[method], phecode_occurrences_path,
                                                           method_formula, negative_weights, phe)
                weights.append(weight_result)

        weights_df = pl.concat(weights)
        return weights_df
