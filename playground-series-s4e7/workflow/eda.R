library(data.table)
library(tidyverse)
library(arrow)
library(pbmcapply)

# Load data ----
train_df <-
  data.table::fread("data/train.csv") |>
  as_tibble()

# Explore ----
glimpse(train_df)

response_counts <-
  count(train_df, Response)

# Create bootstraps ----
create_boots <- function(bid) {
  set.seed(bid)

  train_df |>
    group_by(Response) |>
    sample_n(size = min(response_counts$n) * 0.25) |>
    ungroup() |>
    mutate(Bootstrap = bid, .before = 1)
}

## Run in parrallel
boots <-
  pbmclapply(
    1:10,
    create_boots,
    mc.cores = "10"
  )

boots_df <- collapse::unlist2d(boots, idcols = FALSE)
glimpse(boots_df)

## Save as parquet
fs::dir_create("data/bootstraps")

arrow::write_dataset(
  boots_df,
  path = "data/bootstraps",
  format = "parquet",
  partitioning = "Bootstrap"
)

# EDA ----
boots <-
  arrow::open_dataset("data/bootstraps")

boots |>
  filter(Bootstrap == 1) |>
  collect() |>
  skimr::skim()
