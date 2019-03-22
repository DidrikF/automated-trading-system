"""
In this script samples (based on SEP and SF1 data) is combined together into a final complete dataset, ready for ML.
Some samples might have to be dropped due to missing data in SF1_ART or SF1_ARQ.

The most recent row from sep_featured.csv, and the most recent SF1 row based on datekey is added to sep_sampled.csv.
- If SF1 row is too old compared to the sample date, the sample is dropped.
- If too much fabricated data was used in the generation of the SF1 features, the samples using this row
  is dropped.

"""