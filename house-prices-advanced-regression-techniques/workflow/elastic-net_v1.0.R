library(data.table)
library(tidyverse)
library(tidymodels)

# Load data ----
train_df <-
  data.table::fread("data/train.csv") |>
  as_tibble() |>
  # select_if(~ mean(is.na(.x)) == 0) |>
  mutate(SalePrice = log10(SalePrice))

test_df <-
  data.table::fread("data/test.csv") |>
  as_tibble()

# EDA ----
skimr::skim(train_df)

# Model setup -------------------------------------------------------------
# Elastic Net Model Spec
tune_spec <-
  linear_reg(
    penalty = tune(),
    mixture = tune()
  ) |>
  set_engine("glmnet") |>
  set_mode("regression")

# Parameters grid
set.seed(123)
ff_grid <-
  grid_latin_hypercube(
    extract_parameter_set_dials(tune_spec),
    size = 50
  )

houses_metrics <- metric_set(rmse)

# Model ----
## Cross-validation setup
set.seed(1234)
df_folds <-
  vfold_cv(
    train_df,
    strata = SalePrice,
    v = 10L,
    repeats = 1L
  )

## Recipe
net_rec <-
  recipe(SalePrice ~ ., data = train_df) |>
  update_role(
    c(Id),
    new_role = "id"
  ) |>
  step_zv(all_predictors()) |>
  step_YeoJohnson(all_numeric_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_unknown(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors())

## Combine recipe and model in workflow
net_wf <-
  workflow() |>
  add_recipe(net_rec) |>
  add_model(tune_spec)

doParallel::registerDoParallel(cores = 15)
## Tune Grid (trying a variety of values of Lambda penalty)
tune_output <-
  tune_grid(
    net_wf, # workflow
    resamples = df_folds, # cv folds
    metrics = houses_metrics,
    control = control_resamples(
      save_pred = TRUE
    ),
    grid = ff_grid # penalty grid defined above
  )

## Select best tuning parameters based on OSE
show_best(tune_output)

ose <-
  select_by_one_std_err(
    tune_output,
    metric = "rmse",
    desc(penalty)
  )

## Finalize workflow
final_model <-
  finalize_workflow(x = net_wf, parameters = ose)
final_model_fit <-
  fit(final_model, train_df)

## Explore model
tidy(final_model_fit) |>
  filter(estimate != 0) |>
  arrange(-abs(estimate))

# Predict ----
test_pred <-
  augment(final_model_fit, test_df)

test_pred |>
  filter(is.na(.pred))

# Save ----
test_pred |>
  transmute(
    Id,
    SalePrice = 10^.pred
  ) |>
  data.table::fwrite("results/elastic_net_v1.0.csv")
