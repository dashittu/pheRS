import os
import polars as pl
import sys

from src.PheRS import utils


def map_disease_to_phecode(data_version, disease_hpo_map_file, hpo_phecode_map_file, output_file_name=None):
    """
    Maps diseases to phecodes based on their association with HPO terms.

    Parameters:
    - disease_hpo_map_file: File path to the CSV file containing disease to HPO term mappings.
      Must have columns 'disease_id' and 'hpo_term_id'.
    - hpo_phecode_map_file: File path to the CSV file containing HPO term to phecode mappings.
      Must have columns 'hpo_term_id' and 'phecode'.

    Returns:
    - A pandas.DataFrame with unique mappings from 'disease_id' to 'phecode'.
    """

    # load phecode mapping file by None keyword or custom path
    phemap_dir = os.path.dirname(__file__)
    final_file_path = os.path.join(phemap_dir, "data_raw")
    disease_hpo_path_suffix = ""
    hpo_phecode_path_suffix = ""

    if data_version is None:
        disease_hpo_path_suffix = "disease_hpo_map_omim.csv"
        hpo_phecode_path_suffix = "HPO_phecode_map.csv"
    elif data_version == "custom":
        if disease_hpo_map_file is None or hpo_phecode_map_file is None:
            print("Either disease_hpo_map_file or hpo_phecode_map_file path is None. "
                  "Please provide path for custom data")
            sys.exit(0)
    else:
        print("Invalid data_version. Available data_version value is either None or custom.")
        sys.exit(0)
    if disease_hpo_map_file is None or hpo_phecode_map_file is None:
        disease_hpo_path = os.path.join(final_file_path, disease_hpo_path_suffix)
        hpo_phecode_path = os.path.join(final_file_path, hpo_phecode_path_suffix)
    else:
        disease_hpo_path = disease_hpo_map_file
        hpo_phecode_path = hpo_phecode_map_file

    # Read CSV files into Polars DataFrames
    disease_hpo_df = pl.read_csv(disease_hpo_path,
                                 dtypes={"disease_id": pl.Float64,
                                         "hpo_term_id": str,
                                         "disease_name": str,
                                         "hpo_term_name": str})

    hpo_phecode_df = pl.read_csv(hpo_phecode_path,
                                 dtypes={"hpo_term_id": str,
                                         "hpo_term_name": str,
                                         "phecode": pl.Float64,
                                         "phecode_name": str})

    # Validate input dataframes
    for df, required_cols in zip([disease_hpo_df, hpo_phecode_df],
                                 [['disease_id', 'hpo_term_id'], ['hpo_term_id', 'phecode']]):
        if not set(required_cols).issubset(df.columns):
            raise ValueError(f"DataFrame is missing one of the required columns: {required_cols}")
        if df.select([pl.col(col).is_null().any() for col in required_cols]).to_series().any():
            raise ValueError("Input DataFrame contains missing values in the required columns.")

    # Merge the two dataframes on 'hpo_term_id'
    disease_phecode_map = disease_hpo_df.join(hpo_phecode_df, on='hpo_term_id', how='inner')

    # Drop duplicates to ensure uniqueness and select only the required columns
    disease_phecode_map = disease_phecode_map.select(['disease_id', 'phecode']).unique()

    # Optionally, you could sort the DataFrame for easier readability
    disease_phecode_map = disease_phecode_map.sort(by=['disease_id', 'phecode'])

    # report result
    utils.report_result(disease_phecode_map, phecode_version=None, placeholder="disease_phecode_map",
                        output_file_name=output_file_name)


# if __name__ == "__main__":
#    map_disease_to_phecode(data_version=None, disease_hpo_map_file=None,
#                           hpo_phecode_map_file=None, output_file_name="output.csv")
