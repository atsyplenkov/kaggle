library(tidyverse)
library(tidymodels)
library(pbmcapply)
library(arrow)
library(tictoc)

# Open dataset ----
boots <-
    arrow::open_dataset("data/bootstraps")

# Bootstrap 1 ----
b1 <-
    boots |>
    filter(Bootstrap == 1) |>
    collect() |>
    mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

## Splits
set.seed(1)
b1_split <-
    b1 |>
    dplyr::select(-Bootstrap) |>
    initial_split(
        strata = Response,
        prop = 3 / 4
    )

## Training and testing datasets
b1_train <- training(b1_split)
b1_val <- testing(b1_split)

# Model setup ----
# Elastic Net Model Spec
xgb_spec <-
    boost_tree(
        trees = 1000,
        tree_depth = tune(), min_n = tune(),
        loss_reduction = tune(), ## first three: model complexity
        sample_size = tune(), mtry = tune(), ## randomness
        learn_rate = tune() ## step size
    ) %>%
    set_engine("xgboost") %>%
    set_mode("classification")

# Parameters grid
set.seed(123)
xgb_grid <-
    grid_latin_hypercube(
        tree_depth(),
        min_n(),
        loss_reduction(),
        sample_size = sample_prop(),
        finalize(mtry(), b1_train),
        learn_rate(),
        size = 30
    )

## Metrics to evaluate the model
cv_metrics <- metric_set(roc_auc, f_meas)

# Workflow ----
## Cross-validation setup
set.seed(1)
b1_folds <-
    vfold_cv(
        b1_val,
        strata = Response,
        v = 10L,
        repeats = 1L
    )

## Recipe
b1_recipe <-
    recipe(Response ~ ., data = b1_train) |>
    update_role(
        c(id),
        new_role = "id"
    ) |>
    step_zv(all_predictors()) |>
    step_YeoJohnson(all_numeric()) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors())

## Combine recipe and model in workflow
b1_wf <-
    workflow() |>
    add_recipe(b1_recipe) |>
    add_model(xgb_spec)

## Tune Grid (trying a variety of values of Lambda penalty)
doParallel::registerDoParallel(cores = 10)
tic()
b1_tune <-
    tune_grid(
        b1_wf, # workflow
        resamples = b1_folds, # cv folds
        metrics = metric_set(roc_auc),
        grid = xgb_grid, # penalty grid defined above
        control = control_grid(save_pred = TRUE)
    )
toc()
#> 3700 sec

# Save tunning ----
xg_bundle <- bundle::bundle(b1_tune)
fs::dir_create("models")
qs::qsave(xg_bundle, "models/xg_tuning_bundle.qs")

rm(xg_bundle)
gc()

# Explore tuning ----
## Select best tuning parameters based on OSE
show_best(b1_tune, metric = "roc_auc")

autoplot(b1_tune)

collect_metrics(b1_tune)

best_auc <- select_best(b1_tune, metric = "roc_auc")
best_auc

## Finalize workflow
b1_model <-
    finalize_workflow(x = b1_wf, parameters = best_auc)

## Save model wf
b1_model |>
    bundle::bundle() |>
    qs::qsave("models/xg_tuned_wf.qs")

# Fit model to the training set ----
b1_fit_train <-
    fit(b1_model, b1_train)

b1_train_metrics <-
    augment(b1_fit_train, b1_train, type = "prob")

roc_auc(b1_train_metrics, Response, .pred_1)
#> 0.880

## save model
bundle::bundle(b1_fit_train) |>
    qs::qsave("models/xg_fit_b1_train.qs")

# Fit model to bootstrap 2 ----
b2 <-
    boots |>
    filter(Bootstrap == 2) |>
    collect() |>
    mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

doParallel::registerDoParallel(cores = 10)
b2_fit <-
    fit(b1_model, b2)

## Test models on B2
b2_metrics <-
    augment(b2_fit, b2, type = "prob")

roc_auc(b2_metrics, Response, .pred_1)
#> 0.880

## Test model trained on B1 on B2
b2_metrics_b1model <-
    augment(b1_fit_train, b2, type = "prob")
roc_auc(b2_metrics_b1model, Response, .pred_1)
#> 0.877

## Test model trained on B2 on B1
b1_metrics_b2model <-
    augment(b2_fit, b1, type = "prob")
roc_auc(b1_metrics_b2model, Response, .pred_1)
#> 0.877

## save model
bundle::bundle(b2_fit) |>
    qs::qsave("models/xg_fit_b2.qs")

## Clean memory
rm(
    b1_folds, b1_tune, b2_metrics_b1model,
    b2_metrics, b1_metrics_b2model, b1_split,
    b1, b2,
    b1_train_metrics
)
gc()

# Test dataframe ----
library(data.table)
## Load it
test_df <- data.table::fread("data/test.csv")

## Resave it in parquet split by partitions
test_len <- nrow(test_df)
## Split it in 20 parts
n_parts <- 20
test_len / n_parts

## Assign parts number
test_df[, part := cut(seq(1, .N), breaks = n_parts, labels = FALSE)]

## Save as parquet
fs::dir_create("data/test")
arrow::write_dataset(
    test_df,
    path = "data/test",
    format = "parquet",
    partitioning = "part"
)

rm(test_df)
gc()

# Predict on test dataframe ----
b2_fit <- qs::qread("models/xg_fit_b2.qs") |>
    bundle::unbundle()

test_arrow <-
    open_dataset("data/test")

fs::dir_create("results/xg_b2")

pred_arrow <-
    function(key) {
        key <- as.integer(key)
        key_dir <-
            paste0(
                "results/xg_b2/",
                "part=", key
            )

        key_file <-
            paste0(
                "results/xg_b2/",
                "part=", key,
                "/part-",
                0,
                ".parquet"
            )

        df_key <-
            test_arrow |>
            filter(part == key) |>
            select(-part) |>
            collect() |>
            as.data.table()

        key_res <-
            data.table::data.table(
                id = df_key$id,
                part = key,
                Response = predict(
                    b2_fit,
                    type = "prob",
                    new_data = df_key
                )$.pred_1
            )

        fs::dir_create(key_dir)

        arrow::write_parquet(key_res, key_file)
        rm(key_res, df_key)
        gc()
    }

# tic()
# pred_arrow(2)
# toc()

lapply(
    seq_len(20),
    pred_arrow
)

all_test <-
    arrow::open_dataset("results/xg_b2/") |>
    select(-part)

# Check that there is no NA's
nrow(all_test) == 7669866

arrow::write_parquet(all_test, "results/xgboost_v1.0.parquet")
