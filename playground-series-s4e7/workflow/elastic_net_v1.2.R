library(tidyverse)
library(tidymodels)
library(pbmcapply)
library(arrow)
library(tictoc)

# Open dataset ----
boots <-
  arrow::open_dataset("data/bootstraps")

# Model setup ----
# Elastic Net Model Spec
tune_spec <-
  logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) |>
  set_engine("glmnet") |>
  set_mode("classification")

# Parameters grid
set.seed(123)
ff_grid <-
  grid_latin_hypercube(
    extract_parameter_set_dials(tune_spec),
    size = 50
  )

## Metrics to evaluate the model
cv_metrics <- metric_set(roc_auc, f_meas)

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
  add_model(tune_spec)

## Tune Grid (trying a variety of values of Lambda penalty)
doParallel::registerDoParallel(cores = 10)
tic()
b1_tune <-
  tune_grid(
    b1_wf, # workflow
    resamples = b1_folds, # cv folds
    metrics = metric_set(roc_auc),
    grid = ff_grid # penalty grid defined above
  )
toc()
#> 371.402 sec

## Select best tuning parameters based on OSE
show_best(b1_tune, metric = "roc_auc")

b1_ose <-
  select_by_one_std_err(
    b1_tune,
    metric = "roc_auc",
    desc(penalty)
  )

## Finalize workflow
b1_model <-
  finalize_workflow(x = b1_wf, parameters = b1_ose) |>
  fit(b1_train)

# Bootstrap 2 ----
b2 <-
  boots |>
  filter(Bootstrap == 1) |>
  collect() |>
  mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

## Splits
set.seed(1)
b2_split <-
  b2 |>
  dplyr::select(-Bootstrap) |>
  initial_split(
    strata = Response,
    prop = 3 / 4
  )

## Training and testing datasets
b2_train <- training(b2_split)
b2_val <- testing(b2_split)

## Cross-validation setup
set.seed(1)
b2_folds <-
  vfold_cv(
    b2_val,
    strata = Response,
    v = 3L,
    repeats = 3L
  )

## Recipe
b2_recipe <-
  recipe(Response ~ ., data = b2_train) |>
  update_role(
    c(id),
    new_role = "id"
  ) |>
  step_zv(all_predictors()) |>
  step_YeoJohnson(all_numeric()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors())

## Combine recipe and model in workflow
b2_wf <-
  workflow() |>
  add_recipe(b2_recipe) |>
  add_model(tune_spec)

## Tune Grid (trying a variety of values of Lambda penalty)
doParallel::registerDoParallel(cores = 10)
tic()
b2_tune <-
  tune_grid(
    b2_wf, # workflow
    resamples = b2_folds, # cv folds
    metrics = metric_set(roc_auc),
    grid = ff_grid # penalty grid defined above
  )
toc()
#> 202.438  sec elapsed

## Select best tuning parameters based on OSE
show_best(b2_tune, metric = "roc_auc")

b2_ose <-
  select_by_one_std_err(
    b2_tune,
    metric = "roc_auc",
    desc(penalty)
  )

## Finalize workflow
b2_model <-
  finalize_workflow(x = b2_wf, parameters = b2_ose) |>
  fit(b2_train)

# Compare CV ----
b1_train_probs <-
  predict(b1_model, type = "prob", new_data = b1_train) |>
  bind_cols(predict(b1_model, new_data = b1_train)) |>
  bind_cols(obs = b1_train$Response)

b2_train_probs <-
  predict(b2_model, type = "prob", new_data = b2_train) |>
  bind_cols(predict(b2_model, new_data = b2_train)) |>
  bind_cols(obs = b2_train$Response)

## Looks like 10-fold CV a slightly better than 3-fold repeated 3 times
bind_rows(
  cv_metrics(b1_train_probs, obs, .pred_1, estimate = .pred_class),
  cv_metrics(b2_train_probs, obs, .pred_1, estimate = .pred_class)
)

# Define which bootstrap is the best ----
elastic_model_boot <-
  function(boot_id) {
    ## Load dataset
    b_boots <-
      boots |>
      filter(Bootstrap == boot_id) |>
      collect() |>
      mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

    ## Splits
    set.seed(boot_id)
    df_split <-
      b_boots |>
      dplyr::select(-Bootstrap) |>
      initial_split(
        strata = Response,
        prop = 3 / 4
      )

    ## Training and testing datasets
    df_train <- training(df_split)
    df_val <- testing(df_split)

    ## Cross-validation setup
    set.seed(boot_id)
    df_folds <-
      vfold_cv(
        df_val,
        strata = Response,
        v = 10L,
        repeats = 1L
      )

    ## Recipe
    net_rec <-
      recipe(Response ~ ., data = df_train) |>
      update_role(
        c(id),
        new_role = "id"
      ) |>
      step_zv(all_predictors()) |>
      step_YeoJohnson(all_numeric()) |>
      step_novel(all_nominal_predictors()) |>
      step_dummy(all_nominal_predictors())

    ## Combine recipe and model in workflow
    net_wf <-
      workflow() |>
      add_recipe(net_rec) |>
      add_model(tune_spec)

    doParallel::registerDoParallel(cores = 10)
    ## Tune Grid (trying a variety of values of Lambda penalty)
    tune_output <-
      tune_grid(
        net_wf, # workflow
        resamples = df_folds, # cv folds
        metrics = metric_set(roc_auc),
        grid = ff_grid # penalty grid defined above
      )

    ## Select best tuning parameters based on OSE
    ose <-
      select_by_one_std_err(
        tune_output,
        metric = "roc_auc",
        desc(penalty)
      )

    ## Finalize workflow
    final_model <-
      finalize_workflow(x = net_wf, parameters = ose)
    final_model_fit <-
      fit(final_model, df_train)

    ## Test
    test_probs <-
      predict(final_model_fit, type = "prob", new_data = df_train) |>
      bind_cols(predict(final_model_fit, new_data = df_train)) |>
      bind_cols(obs = df_train$Response)

    ## Test metrics
    response_metrics <-
      cv_metrics(test_probs, obs, .pred_1, estimate = .pred_class)

    ## Return
    ret <-
      tibble(
        Bootstrap = boot_id,
        .metrics = list(response_metrics)
      )

    ## Clean
    rm(tune_output, df_folds, final_model, final_model_fit, df_split)
    gc()

    ## Return
    ret
  }

# Test on 1 bootstrap ----
tictoc::tic()
b1_res <- elastic_model_boot(1)
tictoc::toc()

# Apply to all bootstraps ----
b_res <-
  lapply(
    2:10,
    elastic_model_boot
  )

b_res_all <-
  bind_rows(b1_res, b_res)

b_res_all |>
  unnest(c(.metrics)) |>
  filter(.metric == "roc_auc") |>
  filter(.estimate == max(.estimate))
# 1st bootstrap is the best

# Fit model to the first bootstrap ----
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
  add_model(tune_spec)

## Tune Grid (trying a variety of values of Lambda penalty)
doParallel::registerDoParallel(cores = 10)
tic()
b1_tune <-
  tune_grid(
    b1_wf, # workflow
    resamples = b1_folds, # cv folds
    metrics = metric_set(roc_auc),
    grid = ff_grid # penalty grid defined above
  )
toc()
#> 371.402 sec

b1_ose <-
  select_by_one_std_err(
    b1_tune,
    metric = "roc_auc",
    desc(penalty)
  )

## Finalize workflow
b1_model <-
  finalize_workflow(x = b1_wf, parameters = b1_ose) |>
  fit(b1_train)

## Test
b1_probs <-
  predict(b1_model, type = "prob", new_data = b1_train) |>
  bind_cols(predict(b1_model, new_data = b1_train)) |>
  bind_cols(obs = b1_train$Response)

## Test metrics
b1_metrics <-
  cv_metrics(b1_probs, obs, .pred_1, estimate = .pred_class)

b1_metrics$.estimate

# Predict on test dataset ----
test_df <- data.table::fread("data/test.csv")

doParallel::registerDoParallel(cores = 10)
tictoc::tic()
test_res <-
  data.table::data.table(
    id = test_df$id,
    Response = predict(
      b1_model,
      type = "prob",
      new_data = test_df
    )$.pred_1
  )
tictoc::toc()

# Check that there is no NA's
sum(!is.na(test_res$Response)) == 7669866

# Save ----
fs::dir_create("results")
arrow::write_parquet(test_res, "results/elastic_net_v1.2.parquet")
