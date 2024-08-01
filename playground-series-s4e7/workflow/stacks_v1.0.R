library(tidyverse)
library(tidymodels)
library(pbmcapply)
library(arrow)
library(stacks)
library(tictoc)

parallel::detectCores()

# Open dataset ----
boots <-
  arrow::open_dataset("data/bootstraps")

# Select bootstrap ----
b1 <-
  boots |>
  filter(Bootstrap == 1) |>
  collect() |>
  mutate(Response = factor(Response, levels = c(1, 0), labels = c(1, 0)))

## Splits
set.seed(20240729)
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
set.seed(20240729)
b1_folds <-
  vfold_cv(
    b1_val,
    strata = Response,
    v = 3L,
    repeats = 1L
  )

# Recipe ----
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

b1_wf <-
  workflow() %>%
  add_recipe(b1_recipe)

ctrl_grid <- control_stack_grid()

# Random forest setup ----
rand_forest_spec <-
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 500
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rand_forest_wflow <-
  b1_wf %>%
  add_model(rand_forest_spec)

## Tune RF model
doParallel::registerDoParallel(cores = 10)
tic()
rand_forest_res <-
  tune_grid(
    object = rand_forest_wflow,
    resamples = b1_folds,
    grid = 10,
    control = ctrl_grid
  )
toc()
#> 1357.703 sec elapsed

# Elastic net model setup ----
elastic_net_spec <-
  logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) |>
  set_engine("glmnet") |>
  set_mode("classification")

elastic_net_wflow <-
  b1_wf %>%
  add_model(elastic_net_spec)

## Parameters grid
set.seed(20240729)
elastic_net_grid <-
  grid_latin_hypercube(
    extract_parameter_set_dials(elastic_net_spec),
    size = 30
  )

## Tune Elastic Net model
doParallel::registerDoParallel(cores = 10)
tic()
elastic_net_res <-
  tune_grid(
    object = elastic_net_wflow,
    resamples = b1_folds,
    grid = elastic_net_grid,
    control = ctrl_grid
  )
toc()
#> 58.709 sec elapsed

# Stacks ----
doParallel::registerDoParallel(cores = 10)
tic()
model_st <-
  # initialize the stack
  stacks() %>%
  # add candidate members
  add_candidates(rand_forest_res) %>%
  add_candidates(elastic_net_res) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()
toc()

model_st
autoplot(model_st)
