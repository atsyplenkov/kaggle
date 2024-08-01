# An example of using GPU-accelerated tree building algorithms
#
# NOTE: it can only run if you have a CUDA-enable GPU and the package was
#       specially compiled with GPU support.
#
# For the current functionality, see
# https://xgboost.readthedocs.io/en/latest/gpu/index.html
#

library("xgboost")

# Simulate N x p random matrix with some binomial response dependent on pp columns
set.seed(111)
N <- 1000000
p <- 50
pp <- 25
X <- matrix(runif(N * p), ncol = p)
betas <- 2 * runif(pp) - 1
sel <- sort(sample(p, pp))
m <- X[, sel] %*% betas - 1 + rnorm(N)
y <- rbinom(N, 1, plogis(m))

tr <- sample.int(N, N * 0.75)
dtrain <- xgb.DMatrix(X[tr, ], label = y[tr])
dtest <- xgb.DMatrix(X[-tr, ], label = y[-tr])
evals <- list(train = dtrain, test = dtest)

# An example of running 'gpu_hist' algorithm
# which is
# - similar to the 'hist'
# - the fastest option for moderately large datasets
# - current limitations: max_depth < 16, does not implement guided loss
# You can use tree_method = 'gpu_hist' for another GPU accelerated algorithm,
# which is slower, more memory-hungry, but does not use binning.
param <- list(
    objective = "reg:logistic",
    eval_metric = "auc",
    nthread = 4,
    tree_method = "hist",
    device = "cuda"
)
pt <- proc.time()
bst_gpu <- xgb.train(param, dtrain, evals = evals, nrounds = 50)
proc.time() - pt

eval_log <- bst_gpu$evaluation_log
print(eval_log)

# Compare to the 'hist' algorithm:
param$tree_method <- "hist"
pt <- proc.time()
bst_hist <- xgb.train(param, dtrain, evals = evals, nrounds = 50)
proc.time() - pt



# Load necessary libraries
library(xgboost)

# Example data
data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")

class(train$label)

# Prepare data
train <- agaricus.train
test <- agaricus.test
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
dtest <- xgb.DMatrix(data = test$data, label = test$label)

# Parameters for xgboost
params <- list(
    objective = "binary:logistic",
    eval_metric = "error",
    tree_method = "hist",
    device = "cuda"
)

# Watchlist for evaluation
watchlist <- list(train = dtrain, test = dtest)

# Training the model
model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 1000,
    watchlist = watchlist,
)

eval_log <- model$evaluation_log
print(eval_log)


# section ----
library(xgboost)
library(Matrix)
library(dplyr)
library(pROC)

# Simulate N x p random matrix with some binomial response dependent on pp columns
library(xgboost)
library(Matrix)
library(dplyr)
library(pROC)

# Simulate N x p random matrix with some binomial response dependent on pp columns
set.seed(111)
N <- 1000000
p <- 50
pp <- 25
X <- matrix(runif(N * p), ncol = p)
betas <- 2 * runif(pp) - 1
sel <- sort(sample(p, pp))
m <- X[, sel] %*% betas - 1 + rnorm(N)
y <- rbinom(N, 1, plogis(m))

# Split data
tr <- sample.int(N, N * 0.75)
dtrain <- xgb.DMatrix(X[tr, ], label = y[tr])
dtest <- xgb.DMatrix(X[-tr, ], label = y[-tr])

# Define a grid of hyperparameters
param_grid <- expand.grid(
  tree_depth = c(3, 6, 10),
  min_n = c(1, 5, 10),
  loss_reduction = c(0, 1, 5),
  sample_size = c(0.5, 0.8, 1.0),
  mtry = c(10, 20, 30), # Example mtry values
  learn_rate = c(0.01, 0.1, 0.2)
)

# Function to train and evaluate model
evaluate_model <- function(params, dtrain, dtest) {
  # Print the parameters to debug
  print(params)

  # Train the model
  model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = params$tree_depth,
      min_child_weight = params$min_n,
      gamma = params$loss_reduction,
      subsample = params$sample_size,
      colsample_bytree = params$mtry / ncol(as.matrix(dtrain)), # Example scaling
      eta = params$learn_rate,
      tree_method = "hist",
      device = "cuda"
    ),
    data = dtrain,
    nrounds = 50
  )

  # Predict on test set
  preds <- predict(model, dtest)

  # Extract true labels
  labels <- getinfo(dtest, "label")

  # Calculate AUC
  auc_value <- pROC::auc(labels, preds)
  return(auc_value)
}

# Example of running grid search
results <- lapply(1:nrow(param_grid), function(i) {
  params <- param_grid[i, ]
  # Ensure params are converted to a list
  params <- as.list(params)
  # Evaluate model and return AUC
  tryCatch({
    evaluate_model(params, dtrain, dtest)
  }, error = function(e) {
    cat("Error in evaluation:", e$message, "\n")
    return(NA)  # Return NA if there's an error
  })
})

# Convert results to a numeric vector
results <- unlist(results)

# Display results
print(results)

