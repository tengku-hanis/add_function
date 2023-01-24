# Additional metrics

# Additional function for alarm ----
beep <- function(n = 10){
  for(i in seq(n)){
    system("rundll32 user32.dll,MessageBeep -1")
    Sys.sleep(.5)
  }
}

# Specificity ----
metric_specificity <- new_metric_class(
  classname = "specificity",
  
  initialize = function(name = "specificity", ...) {
    super$initialize(name = name, ...)
    self$true_negative_sum <- self$add_weight(name = "true_negative_sum",
                                              initializer = "zeros",
                                              dtype = "float32")
    self$false_positive_sum <- self$add_weight(name = "false_positive_sum",
                                               initializer = "zeros",
                                               dtype = "float32")
    self$total_samples <- self$add_weight(name = "total_samples",
                                          initializer = "zeros",
                                          dtype = "int32")
  },
  
  update_state = function(y_true, y_pred, sample_weight = NULL) {
    num_samples <- tf$shape(y_pred)[1]
    y_pred <- k_greater_equal(y_pred, 0.5)
    y_pred <- k_cast(y_pred, dtype = "float32")
    true_neg <- k_sum(k_round(k_clip((1 - y_true) * (1 - y_pred), 0, 1)))
    false_pos <- k_sum(k_round(k_clip((1 - y_true) * y_pred, 0, 1)))
    self$true_negative_sum$assign_add(true_neg)
    self$false_positive_sum$assign_add(false_pos)
    self$total_samples$assign_add(num_samples)
  },
  
  result = function() {
    true_neg <- self$true_negative_sum
    false_pos <- self$false_positive_sum
    specificity <- true_neg / (true_neg + false_pos + k_epsilon())
    specificity
  },
  
  reset_state = function() {
    self$true_negative_sum$assign(0)
    self$false_positive_sum$assign(0)
    self$total_samples$assign(0L)
  }
)

# Youlden J index ----
metric_youlden_j_index <- new_metric_class(
  classname = "youldenJIndex",
  
  initialize = function(name = "youlden_j_index", ...) {
    super$initialize(name = name, ...)
    self$true_positive_sum <- self$add_weight(name = "true_positive_sum",
                                              initializer = "zeros",
                                              dtype = "float32")
    self$false_positive_sum <- self$add_weight(name = "false_positive_sum",
                                               initializer = "zeros",
                                               dtype = "float32")
    self$false_negative_sum <- self$add_weight(name = "false_negative_sum",
                                               initializer = "zeros",
                                               dtype = "float32")
    self$true_negative_sum <- self$add_weight(name = "true_negative_sum",
                                              initializer = "zeros",
                                              dtype = "float32")
  },
  
  update_state = function(y_true, y_pred, sample_weight = NULL) {
    num_samples <- tf$shape(y_pred)[1]
    y_pred <- k_greater_equal(y_pred, 0.5)
    y_pred <- k_cast(y_pred, dtype = "float32")
    true_pos <- k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
    false_pos <- k_sum(k_round(k_clip((1 - y_true) * y_pred, 0, 1)))
    false_neg <- k_sum(k_round(k_clip(y_true * (1-y_pred), 0, 1)))
    true_neg <- k_sum(k_round(k_clip((1 - y_true) * (1 - y_pred), 0, 1)))
    self$true_positive_sum$assign_add(true_pos)
    self$false_positive_sum$assign_add(false_pos)
    self$false_negative_sum$assign_add(false_neg)
    self$true_negative_sum$assign_add(true_neg)
  },
  
  result = function() {
    true_pos <- self$true_positive_sum
    false_pos <- self$false_positive_sum
    false_neg <- self$false_negative_sum
    true_neg <- self$true_negative_sum
    youlden_j_index <- (true_pos / (true_pos + false_neg)) + (true_neg / (true_neg + false_pos)) - 1
    youlden_j_index
  },
  
  reset_state = function() {
    self$true_positive_sum$assign(0)
    self$false_positive_sum$assign(0)
    self$false_negative_sum$assign(0)
    self$true_negative_sum$assign(0)
  }
)

# F1 score ----
metric_f1_score <- new_metric_class(
  classname = "f1score",
  
  initialize = function(name = "f1_score", ...) {
    super$initialize(name = name, ...)
    self$true_positive_sum <- self$add_weight(name = "true_positive_sum",
                                              initializer = "zeros",
                                              dtype = "float32")
    self$false_positive_sum <- self$add_weight(name = "false_positive_sum",
                                               initializer = "zeros",
                                               dtype = "float32")
    self$false_negative_sum <- self$add_weight(name = "false_negative_sum",
                                               initializer = "zeros",
                                               dtype = "float32")
  },
  
  update_state = function(y_true, y_pred, sample_weight = NULL) {
    y_pred <- k_greater_equal(y_pred, 0.5)
    y_pred <- k_cast(y_pred, dtype = "float32")
    true_pos <- k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
    false_pos <- k_sum(k_round(k_clip((1 - y_true) * y_pred, 0, 1)))
    false_neg <- k_sum(k_round(k_clip(y_true * (1-y_pred), 0, 1)))
    self$true_positive_sum$assign_add(true_pos)
    self$false_positive_sum$assign_add(false_pos)
    self$false_negative_sum$assign_add(false_neg)
  },
  
  result = function() {
    true_pos <- self$true_positive_sum
    false_pos <- self$false_positive_sum
    false_neg <- self$false_negative_sum
    precision <- true_pos / (true_pos + false_pos)
    recall <- true_pos / (true_pos + false_neg)
    f1_score <- 2*(precision * recall) / (precision + recall)
    f1_score
  },
  
  reset_state = function() {
    self$true_positive_sum$assign(0)
    self$false_positive_sum$assign(0)
    self$false_negative_sum$assign(0)
  }
)

