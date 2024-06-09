import numpy as np
import pandas as pd

np.random.seed(0)  # For reproducibility

read_file = pd.read_csv('/Users/dayoshittu/Downloads/icd_event_sample.csv')

dob_start_date = np.datetime64('1925-01-01')
dob_end_date = np.datetime64('1965-12-31')

read_file['dob'] = pd.to_datetime(
    np.random.randint(dob_start_date.astype(int), dob_end_date.astype(int), size=len(read_file))
    , unit='D')

start_date = np.datetime64('1993-01-01')
end_date = np.datetime64('2014-12-31')

read_file['date'] = pd.to_datetime(np.random.randint(start_date.astype(int), end_date.astype(int), size=len(read_file))
                                   , unit='D')

# Example dictionary with PheCodes and their corresponding ICD versions
icd_codes = {
    "001.0": "ICD9",
    "001.1": "ICD9",
    "003.0": "ICD9",
    "003.1": "ICD9",
    "003.2": "ICD9",
    "004.0": "ICD9",
    "004.1": "ICD9",
    "004.2": "ICD9",
    "004.3": "ICD9",
    "004.8": "ICD9",
    "004.9": "ICD9",
    "007.0": "ICD9",
    "007.1": "ICD9",
    "007.2": "ICD9",
    "007.3": "ICD9",
    "007.4": "ICD9",
    "007.5": "ICD9",
    "007.8": "ICD9",
    "007.9": "ICD9",
    "008.0": "ICD9",
    "A00.0": "ICD10",
    "A00.1": "ICD10",
    "A00.9": "ICD10",
    "A01.0": "ICD10",
    "A01.1": "ICD10",
    "A01.2": "ICD10",
    "A01.3": "ICD10",
    "A01.4": "ICD10",
    "A02.0": "ICD10",
    "A02.1": "ICD10",
    "A02.2": "ICD10",
    "A03.0": "ICD10",
    "A03.1": "ICD10",
    "A03.2": "ICD10",
    "A03.3": "ICD10",
    "A03.8": "ICD10",
    "A03.9": "ICD10",
    "A04.0": "ICD10",
    "A04.1": "ICD10",
    "A04.2": "ICD10",
    "A04.3": "ICD10",
    "A04.4": "ICD10",
    "A04.5": "ICD10",
    "A04.6": "ICD10",
    "A04.7": "ICD10",
    "A04.8": "ICD10",
    "A04.9": "ICD10",
    "A05.0": "ICD10",
    "A05.1": "ICD10"
}

read_file['ICD'] = np.random.choice(list(icd_codes.keys()), size=len(read_file))
read_file['flag'] = read_file['ICD'].map(icd_codes)
read_file['flag'] = read_file['flag'].str.replace('ICD', '')

read_file['occurrence_age'] = ((read_file['date'] - read_file['dob']).dt.days / 365.25).round(2)

# For demonstration purposes, I'll assume all provided PheCodes map to ICD9.
# You will need to verify and adjust these mappings based on accurate medical classification data.
read_file500 = read_file.head(500)

read_file500.to_csv('/Users/dayoshittu/Downloads/icd_data_sample.csv', index=False)

# demographics

demo_file = read_file500.copy()
demo_file['sex'] = np.random.choice(['M', 'F'], size=len(demo_file))

enroll_start_date = np.datetime64('1955-01-01')
enroll_end_date = np.datetime64('1970-12-31')

demo_file['start_date'] = pd.to_datetime(np.random.randint(enroll_start_date.astype(int), enroll_end_date.astype(int),
                                                           size=len(demo_file)), unit='D')

last_start_date = np.datetime64('2016-01-01')
last_end_date = np.datetime64('2023-12-31')

demo_file['last_date'] = pd.to_datetime(np.random.randint(last_start_date.astype(int), last_end_date.astype(int),
                                                          size=len(demo_file)), unit='D')

demo_file = demo_file.drop(columns=['date', 'ICD', 'flag', 'occurrence_age'])
demo_file.to_csv('/Users/dayoshittu/Downloads/demo_data_sample.csv', index=False)
