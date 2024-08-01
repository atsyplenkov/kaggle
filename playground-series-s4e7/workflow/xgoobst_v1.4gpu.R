# Install experimental XGBoost ----
# renv::remove("xgboost")
# renv::install(
#     "/mnt/c/users/tsyplenkova/downloads/xgboost_r_gpu_linux_82d846bbeb83c652a0b1dff0e3519e67569c4a3d.tar.gz"
# )

# https://xgboost.readthedocs.io/en/stable/R-package/xgboostPresentation.html

library(tidyverse)
library(tidymodels)
library(pbmcapply)
library(arrow)
library(xgboost)
library(caret)
library(tictoc)

# Open dataset ----
boots <-
    arrow::open_dataset("data/bootstraps_50pct")

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

## Recipe
b1_recipe <-
    recipe(Response ~ ., data = b1_train) |>
    update_role(
        c(id),
        new_role = "id"
    ) |>
    step_zv(all_predictors()) |>
    step_YeoJohnson(all_numeric()) |>
    step_pca(all_numeric_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors())

rec_xg_prep <- b1_recipe %>% prep()

x_train <- rec_xg_prep %>%
    bake(all_predictors(), new_data = NULL, composition = "dgCMatrix")
y_train <- rec_xg_prep %>%
    bake(all_outcomes(), new_data = NULL) |>
    pull(Response) |>
    as.character() |>
    as.integer()
x_test <- rec_xg_prep %>%
    bake(all_predictors(), new_data = b1_val, composition = "dgCMatrix")
y_test <- rec_xg_prep %>%
    bake(all_outcomes(), new_data = b1_val) |>
    pull(Response) |>
    as.character() |>
    as.integer()

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# Define a grid of hyperparameters
set.seed(3107)
param_grid <-
    expand.grid(
        tree_depth = c(3, 6, 10),
        min_n = c(1, 5, 10),
        loss_reduction = c(0, 1, 5),
        sample_size = c(0.5, 0.8, 1.0),
        mtry = c(8, 10, 12), # Example mtry values
        learn_rate = c(0.01, 0.1, 0.2)
    ) |>
    sample_n(100)

# set.seed(3107)
# param_grid <- grid_latin_hypercube(
#     tree_depth(),
#     min_n(),
#     loss_reduction(),
#     sample_size = sample_prop(),
#     finalize(mtry(), dtrain),
#     learn_rate(),
#     size = 50
# )

# Function to train and evaluate model
evaluate_model <- function(params, dtrain, dtest) {
    # Print the parameters to debug
    model <-
        xgb.train(
            data = dtrain,
            objective = "binary:logistic",
            eval_metric = "auc",
            max_depth = params$tree_depth,
            min_child_weight = params$min_n,
            gamma = params$loss_reduction,
            subsample = params$sample_size,
            colsample_bytree = params$mtry / ncol(dtrain), # Example scaling
            eta = params$learn_rate,
            nrounds = 500,
            watchlist = list(train = dtrain, validate = dtest),
            print_every_n = 500,
            tree_method = "hist",
            device = "cuda",
            nthread = 10
        )

    # Predict on test set
    preds <- predict(model, dtest)
    labels <- getinfo(dtest, "label")

    # Calculate AUC
    auc <- pROC::auc(labels, preds)
    return(auc)
}

# Example of running grid search
tic()
results <- lapply(
    seq_len(nrow(param_grid)),
    function(i) {
        params <- param_grid[i, ]
        evaluate_model(params, dtrain, dtest)
    }
)
toc()

# Display results
max(unlist(results))

# Explore results ----
param_grid |>
    mutate(AUC = unlist(results)) |>
    pivot_longer(
        -AUC,
        values_to = "value",
        names_to = "parameter"
    ) %>%
    ggplot(aes(value, AUC, color = parameter)) +
    geom_point(alpha = 0.8, show.legend = FALSE) +
    facet_wrap(~parameter, scales = "free_x") +
    labs(x = NULL, y = "AUC")

param_grid |>
    mutate(AUC = unlist(results)) |>
    rowid_to_column() |>
    filter(AUC == max(AUC))
# 20

# Fit to B2 ----
b2 <-
    boots |>
    filter(Bootstrap == 2) |>
    collect() |>
    mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

x_fit <- rec_xg_prep %>%
    bake(all_predictors(), new_data = b2, composition = "dgCMatrix")
y_fit <- rec_xg_prep %>%
    bake(all_outcomes(), new_data = b2) |>
    pull(Response) |>
    as.character() |>
    as.integer()

b2_fit <- xgb.DMatrix(data = x_train, label = y_train)

params <- param_grid[59, ]

b2_model <-
    xgb.train(
        data = b2_fit,
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = params$tree_depth,
        min_child_weight = params$min_n,
        gamma = params$loss_reduction,
        subsample = params$sample_size,
        colsample_bytree = params$mtry / ncol(dtrain), # Example scaling
        eta = params$learn_rate,
        nrounds = 2000,
        watchlist = list(train = b2_fit),
        print_every_n = 100,
        tree_method = "hist",
        device = "cuda",
        nthread = 15
    )

# 0.885632

# xgb.save(b2_model, "models/xgboost.model")
b2_model <- xgb.load("models/xgboost.model")

# Predict on TEST dataset ----
test_arrow <-
    open_dataset("data/test")

fs::dir_create("results/xg_b2_gpu2")

test_1 <-
    test_arrow |>
    select(-part) |>
    collect()

x_key <- rec_xg_prep %>%
    bake(all_predictors(), new_data = test_1, composition = "dgCMatrix")

key_matrix <- xgb.DMatrix(data = x_key)
key_preds <- predict(b2_model, x_key, type = "prob")
summary(key_preds)

key_res <-
    data.table::data.table(
        id = test_1$id,
        Response = key_preds
    )

arrow::write_parquet(key_res, "results/xgboost_gpu_v1.4pca.parquet")

pred_gpu <-
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
            collect()

        x_key <- rec_xg_prep %>%
            bake(all_predictors(), new_data = df_key, composition = "dgCMatrix")
        y_key <- rec_xg_prep %>%
            bake(all_outcomes(), new_data = df_key) |>
            pull(Response) |>
            as.character() |>
            as.integer()

        key_matrix <- xgb.DMatrix(data = x_key, label = y_key)

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







mod_xg_1 <- xgb.train(
    data = dtrain,
    objective = "binary:logistic",
    eval_metric = "auc",
    nrounds = 1000,
    subsample = 0.5,
    colsample_bytree = 0.8,
    watchlist = list(train = dtrain, validate = dtest),
    print_every_n = 20,
    tree_method = "hist",
    device = "cuda"
)


# Model setup ----
# Elastic Net Model Spec
xgb_spec <-
    boost_tree(
        trees = 500,
        tree_depth = tune(), min_n = tune(),
        loss_reduction = tune(), ## first three: model complexity
        sample_size = tune(), mtry = tune(), ## randomness
        learn_rate = tune() ## step size
    ) |>
    set_engine(
        "xgboost",
        params = list(tree_method = "hist", device = "cuda")
    ) |>
    set_mode("classification")



## Metrics to evaluate the model
cv_metrics <- metric_set(roc_auc, f_meas)

# Workflow ----
## Cross-validation setup
set.seed(1)
b1_folds <-
    vfold_cv(
        b1_val,
        strata = Response,
        v = 3L,
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
#> 704 sec

# Save tunning ----
xg_bundle <- bundle::bundle(b1_tune)
fs::dir_create("models")
qs::qsave(xg_bundle, "models/xg_tuning_bundle_v1.1.qs")

rm(xg_bundle)
gc()

# Explore tuning ----
## Select best tuning parameters based on OSE
show_best(b1_tune, metric = "roc_auc") |>
    glimpse()

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
    qs::qsave("models/xg_tuned_wf_v1.1.qs")

# Fit model to the training set ----
b1_fit_train <-
    fit(b1_model, b1_train)

b1_train_metrics <-
    augment(b1_fit_train, b1_train, type = "prob")

roc_auc(b1_train_metrics, Response, .pred_1)
#> 0.880

## save model
bundle::bundle(b1_fit_train) |>
    qs::qsave("models/xg_fit_b1_train_v1.1.qs")

# TODO:
# -[] fit model to the second bootstrap
#


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
