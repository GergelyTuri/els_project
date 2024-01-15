# Recorded data structure and description

## Description

Most of the data is located in the lab google dive. The data is organized in the following way (simplified version):

```
|-- TURI_lab
    |-- Data
        |-- PTSD_project
            |-- PTSD_x_ <-cohorts of micedata
                |-- PTSD_shockboxes <- raw videos
                |-- PTSD_x_workbook <- experiment log
                .
                other files and folders
                .
                .
                .
                |-- !analysis <- analysis folder with Colab notebooks and summary data
                    |-- summary presentations <- summary presentations
                    |-- saveFolder <- figures
                    |-- colab notebooks <- notebooks
                    bunch of other files
```

From the "bunch of other files" the most important is the `all_cohorts_freezing_FINAL`, which contains the clean data for all mice recoded and processed.
