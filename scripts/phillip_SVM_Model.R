library(tidyverse)
library(e1071)
library(caret)
library(kernlab)

# Load cleaned data
load('data/claims-clean.RData')

# Check if multiclass exists, otherwise create it from binary
if(!"mclass" %in% names(claims_clean)) {
  # If no multiclass column exists, create a placeholder
  # You'll need to replace this with actual multiclass labels if available
  claims_clean <- claims_clean %>%
    mutate(mclass = bclass)
}

###############################################################################
## BINARY CLASSIFICATION
###############################################################################

# Prepare binary labels
claims_binary <- claims_clean %>%
  mutate(
    bclass_num = if_else(bclass == "Relevant claim content", 1, 0),
    bclass_factor = factor(bclass_num, labels = c("NotClaim", "Claim"))
  )

# Extract features (remove ID and class columns)
drop_cols <- c(".id", "bclass", "bclass_num", "bclass_factor", "mclass")
feature_cols <- setdiff(names(claims_binary), drop_cols)

X_binary <- as.matrix(claims_binary[, feature_cols])
y_binary <- claims_binary$bclass_factor

# Train/test split for evaluation
set.seed(197)
n <- nrow(X_binary)
train_idx <- sample(seq_len(n), size = floor(0.8 * n))
test_idx <- setdiff(seq_len(n), train_idx)

X_train_bin <- X_binary[train_idx, ]
X_test_bin <- X_binary[test_idx, ]
y_train_bin <- y_binary[train_idx]
y_test_bin <- y_binary[test_idx]

# Train SVM with RBF kernel for binary classification
svm_binary <- svm(
  x = X_train_bin,
  y = y_train_bin,
  kernel = "radial",
  cost = 10,
  gamma = 0.01,
  scale = TRUE,
  probability = TRUE
)

# Evaluate on test set
pred_binary_test <- predict(svm_binary, X_test_bin)
accuracy_binary <- mean(pred_binary_test == y_test_bin)

# Print results
print("=== BINARY CLASSIFICATION RESULTS ===")
print(paste("Test Accuracy:", round(accuracy_binary * 100, 2), "%"))
print(confusionMatrix(pred_binary_test, y_test_bin))

# Train final model on ALL data
svm_binary_final <- svm(
  x = X_binary,
  y = y_binary,
  kernel = "radial",
  cost = 10,
  gamma = 0.01,
  scale = TRUE,
  probability = TRUE
)

###############################################################################
## MULTICLASS CLASSIFICATION
###############################################################################

# Prepare multiclass labels
claims_multi <- claims_clean %>%
  mutate(mclass_factor = factor(mclass))

X_multi <- as.matrix(claims_multi[, feature_cols])
y_multi <- claims_multi$mclass_factor

# Train/test split for evaluation
X_train_multi <- X_multi[train_idx, ]
X_test_multi <- X_multi[test_idx, ]
y_train_multi <- y_multi[train_idx]
y_test_multi <- y_multi[test_idx]

# Train SVM with RBF kernel for multiclass classification
svm_multi <- svm(
  x = X_train_multi,
  y = y_train_multi,
  kernel = "radial",
  cost = 10,
  gamma = 0.01,
  scale = TRUE,
  probability = TRUE
)

# Evaluate on test set
pred_multi_test <- predict(svm_multi, X_test_multi)
accuracy_multi <- mean(pred_multi_test == y_test_multi)

print("\n=== MULTICLASS CLASSIFICATION RESULTS ===")
print(paste("Test Accuracy:", round(accuracy_multi * 100, 2), "%"))
print(confusionMatrix(pred_multi_test, y_test_multi))

# Train final model on ALL data
svm_multi_final <- svm(
  x = X_multi,
  y = y_multi,
  kernel = "radial",
  cost = 10,
  gamma = 0.01,
  scale = TRUE,
  probability = TRUE
)

###############################################################################
## SAVE MODELS
###############################################################################

# Save both models
save(svm_binary_final, file = 'results/svm-binary-model.RData')
save(svm_multi_final, file = 'results/svm-multi-model.RData')

# Save feature names for later use
feature_names <- feature_cols
save(feature_names, file = 'results/feature-names.RData')

# Save class label mapping for binary
bclass_levels <- levels(claims_clean$bclass)
save(bclass_levels, file = 'results/bclass-levels.RData')