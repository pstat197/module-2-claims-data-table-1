## PREPROCESSING
#################

# source('scripts/preprocessing.R')
# load('data/claims-raw.RData')
# claims_clean <- claims_raw %>%
#   parse_data()
# save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras3) 
library(tensorflow)
library(reticulate) 

# load cleaned data
load('data/claims-clean-example.RData')

# Create training/testing split
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# Define the text preprocessing layer
# This setup uses bigrams (word pairs) and TF-IDF
preprocess_layer <- layer_text_vectorization(
  standardize = "lower_and_strip_punctuation", 
  split = 'whitespace',
  ngrams = 2,
  max_tokens = 20000, 
  output_mode = 'tf_idf'
)

# Build the vocabulary from the training text
preprocess_layer %>% adapt(train_text)

# Define the NN architecture
# We used the functional API to build the model
text_input <- layer_input(shape = c(1), dtype = "string", name = "text_input")

output <- text_input %>%
  preprocess_layer() %>%
  layer_dense(units = 32) %>% 
  layer_dropout(0.4) %>%      
  layer_dense(units = 16) %>% 
  layer_dropout(0.4) %>%      
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

model <- keras_model(inputs = text_input, outputs = output)

summary(model)

# Configure the model for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.0005), # Use a lower learning rate
  metrics = 'binary_accuracy'
)

# Train the model
history <- model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3, 
      epochs = 4,             
      batch_size = 32)        

plot(history)

## EVALUATE ON TEST SET
################################

test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

test_metrics <- model %>%
  evaluate(test_text, test_labels)


print(test_metrics)
cat(sprintf("Test Set Loss: %.4f\n", test_metrics$loss))
cat(sprintf("Test Set Accuracy: %.4f\n", test_metrics$binary_accuracy))


save_model(model, "results/example-model.keras", overwrite = TRUE)
