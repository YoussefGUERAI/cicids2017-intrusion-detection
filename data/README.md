# Dataset setup

This project uses the machine-learning CSV release of **CICIDS2017**. The raw data are intentionally excluded from Git because the files are large and are maintained by the dataset authors.

1. Download `MachineLearningCSV.zip` from the [official CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Extract the eight CSV files into this directory.
3. Keep the original filenames. The notebooks discover the files deterministically and preserve each source filename for audit purposes.

Expected files include:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

The spelling in the published filenames is preserved, including `Infilteration`.
