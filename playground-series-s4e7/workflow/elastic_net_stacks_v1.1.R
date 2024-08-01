library(tidyverse)
library(tidymodels)
library(pbmcapply)
library(arrow)
library(stacks)
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

# Stacks control
ctrl_grid <- control_stack_grid()

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
  add_model(tune_spec)

## Tune Grid (trying a variety of values of Lambda penalty)
doParallel::registerDoParallel(cores = 10)
tic()
b1_tune <-
  tune_grid(
    b1_wf, # workflow
    resamples = b1_folds, # cv folds
    metrics = metric_set(roc_auc),
    control = ctrl_grid,
    grid = ff_grid # penalty grid defined above
  )
toc()
#> 98.573 sec

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
  filter(Bootstrap == 2) |>
  collect() |>
  mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

## Splits
set.seed(2)
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
set.seed(2)
b2_folds <-
  vfold_cv(
    b2_val,
    strata = Response,
    v = 3L,
    repeats = 1L
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
    control = ctrl_grid,
    grid = ff_grid # penalty grid defined above
  )
toc()
#> 95.206 sec elapsed

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

# Stack bootstraps ----
## https://stacks.tidymodels.org/articles/classification.html

## FAILED
## Since to blend predictions, models should be
## trained on the same dataset
boot_stacks <-
  # initialize the stack
  stacks() %>%
  # add candidate members
  add_candidates(b1_tune) %>%
  add_candidates(b2_tune) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()





# function ----------------------------------------------------------------
elastic_model <-
  function(.df) {
    ## Bootstrap ID
    boot_id <- .df$Bootstrap[1]

    ## Splits
    set.seed(boot_id)
    df_split <-
      .df |>
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
        v = 3L,
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
        control = control_resamples(
          save_pred = TRUE,
          event_level = "second"
        ),
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
      bind_rows(
        roc_auc(test_probs, obs, .pred_1),
        accuracy(test_probs, obs, .pred_class),
        specificity(test_probs, obs, .pred_class),
        sensitivity(test_probs, obs, .pred_class),
        f_meas(test_probs, obs, .pred_class)
      ) |>
      mutate(
        .lambda = ose$penalty,
        .mixture = ose$mixture
      )

    ## Return
    tibble(
      Bootstrap = boot_id,
      .metrics = list(response_metrics)
    )
  }

# Test on 1 bootstrap ----
b1 <-
  boots |>
  filter(Bootstrap == 1) |>
  collect() |>
  mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

tictoc::tic()
b1_res <- elastic_model(b1)
tictoc::toc()

b1_res$test_probs[[1]] |>
  accuracy(obs, .pred_class)
roc_auc(obs, .pred_1, event_level = "second")

# Predict on test dataset ----
test_df <- data.table::fread("data/test.csv")

doParallel::registerDoParallel(cores = 10)
tictoc::tic()
test_res <-
  data.table::data.table(
    id = test_df$id,
    Response = predict(
      final_model_fit,
      type = "prob",
      new_data = test_df
    )$.pred_1
  )
tictoc::toc()

# Check that there is no NA's
sum(!is.na(test_res$Response)) == 7669866

# Save ----
fs::dir_create("results")
data.table::fwrite(
  test_res,
  "results/elastic_net_v1.0.csv"
)

arrow::write_parquet(test_res, "results/elastic_net_v1.0.parquet")
