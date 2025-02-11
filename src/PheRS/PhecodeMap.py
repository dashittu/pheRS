import os
import polars as pl
import sys
# noinspection PyUnresolvedReferences,PyProtectedMember
from pheRS.src.PheRS import queries, utils


class PhecodeMap:
    """
        Class PhecodeMap implements one function
         - get_phecode_occurrences that maps icd_occurrences with the icd_phecode_map
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
                                              schema_overrides={"ICD": str})
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

    def get_phecode_occurrences(self, phecode_map_file_path=None, dx_icd=None,
                                phecode_version="X", icd_version="US", include_occurrence_age=False,
                                phecode_of_interest_path=None, disease_id=None, disease_phecode_map=None,
                                output_file_name=None):
        """
        Maps ICD occurrences to phecodes by merging ICD occurrence data with a mapping table.

        Parameters:
        - icd_occurrences: polars.DataFrame containing ICD code occurrences for each person in the cohort.
            Must have columns 'icd', 'flag', and 'person_id'.
        - icd_phecode_map_path: polars.DataFrame containing the mapping from ICD codes to phecodes.
            Must have columns 'icd', 'flag', and 'phecode'.
        - dx_icd: Optional. polars.DataFrame containing ICD codes to be removed. Must have columns 'icd' and 'flag'.
            If None, no ICD codes are removed.
        - phecode_version: Version of phecode to be merged. Either X or 1.2
        - icd_version: Version of ICD code to be merged. Either US, WHO or custom

        Returns:
        - A polars.DataFrame with columns 'person_id' and 'phecode', representing the occurrence of phecodes per person.
        """

        # One of disease_id or phecode_of_interest_path must be provided
        if disease_id is None and phecode_of_interest_path is None:
            print("Either disease_id or phecode_of_interest_path must be provided. Both cannot be None.")
            sys.exit(0)

        # load phecode mapping file by version or by custom path
        # noinspection PyGlobalUndefined
        # global icd_occurrences
        icd_phecode_map = utils.get_phecode_mapping(
            phecode_version=phecode_version,
            icd_version=icd_version,
            phecode_map_file_path=phecode_map_file_path,
            keep_all_columns=False
        )

        if phecode_of_interest_path is not None:
            print(f"\033[1mFiltering icd_phecode_map by phecodes from {phecode_of_interest_path}...")
            phecode_filter_df = pl.read_csv(
                phecode_of_interest_path,
                schema_overrides={"phecode": str}
            )

            icd_phecode_map = icd_phecode_map.join(
                phecode_filter_df.select(["phecode", "flag"]).unique(),
                on=["phecode", "flag"],
                how="inner"
            )
            print(f"\033[1mDone filtering. Remaining rows in icd_phecode_map: {icd_phecode_map.height}")

        
        elif disease_id is not None:
            if not disease_phecode_map:
                print("A disease_phecode_map must be provided if disease_id is given.")
                sys.exit(0)
    
            print(f"\033[1mMapping disease_id {disease_id} to phecodes from {disease_phecode_map}...")
            disease_phecode_map = pl.read_csv(
                disease_phecode_map,
                schema_overrides={"phecode": str}
            )

            # Validation check
            utils.check_disease_phecode_map(disease_phecode_map)
            
            # Filter for the given disease_id
            disease_phecode_map = disease_phecode_map.filter(pl.col("disease_id") == disease_id)
            # Keep only "phecode"
            disease_phecode_map = disease_phecode_map.select("phecode").unique()
            # Join with icd_phecode_map
            icd_phecode_map = icd_phecode_map.join(
                disease_phecode_map,
                on="phecode",
                how="inner"
            )
            print(f"\033[1mDone mapping disease_id. Remaining rows in icd_phecode_map: {icd_phecode_map.height}")

        # make a copy of self.icd_events
        icd_events = self.icd_events.clone()

        # Required columns
        if include_occurrence_age:
            icd_events = icd_events[["person_id", "ICD", "flag", "occurrence_age"]]
        else:
            icd_events = icd_events[["person_id", "ICD", "flag"]]

        print()
        print(f"\033[1mMapping ICD codes to phecode {phecode_version}...")

        # Validation checks
        utils.check_icd_occurrences(icd_events)
        utils.check_icd_phecode_map(icd_phecode_map)

        # Remove ICD codes if dx_icd is provided
        if dx_icd is not None and not dx_icd.empty:
            dx_icd = pl.read_csv(dx_icd)
            dx_icd = dx_icd.with_columns(pl.lit(1).alias('merge_indicator'))  # Temporary merge indicator

            icd_occurrences = icd_events.join(dx_icd, on=['ICD', 'flag'], how='left')
            icd_events = icd_occurrences.filter(pl.col("merge_indicator").is_null()).drop("merge_indicator")

        # Merge icd occurrences with icd phecode map
        if phecode_version in ["X", "1.2"]:
            phecode_occurrences = icd_events.join(icd_phecode_map, on=['ICD', 'flag'],
                                                  how='inner')
        else:
            phecode_occurrences = pl.DataFrame()

        if include_occurrence_age:
            phecode_occurrences = phecode_occurrences.select(['person_id', 'phecode', 'occurrence_age']).unique()
        else:
            phecode_occurrences = phecode_occurrences.select(['person_id', 'phecode']).unique()

        if not phecode_occurrences.is_empty():
            if include_occurrence_age:
                phecode_occurrences = phecode_occurrences.group_by(
                    ["person_id", "phecode"]).agg([pl.col("occurrence_age").min().alias("occurrence_age"), 
                    pl.count().alias("num_occurrences")])
            else:
                phecode_occurrences = phecode_occurrences.group_by(["person_id", "phecode"]).len().rename(
                    {"len": "num_occurrences"})

        # report result
        utils.report_result(phecode_occurrences, placeholder='phecode_occurrences',
                            output_file_name=output_file_name)
