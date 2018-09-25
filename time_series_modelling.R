
library(forecast)
library(ggplot2)
library(readr)
library(MLmetrics)
library(bimixt)

##################################
# Global Variables and Functions
##################################

results_method = c()
results_model = c()
results_rmse = c()
results_mape = c()
results_mad = c()
results_freq = c()
results_pred = c()
results_err = c()

MAD = function(predicted, actual){
  sum(abs(predicted - actual))/length(predicted)
}

logerrors = function(err, method, frequency){
  return(
    function(err){
      results_method <<- c(results_method, method)
      results_model <<- c(results_model, NA)
      results_rmse <<- c(results_rmse, NA)
      results_mape <<- c(results_mape, NA)
      results_mad <<- c(results_mad, NA)
      results_freq <<- c(results_freq, frequency)
      results_pred <<- c(results_pred, NA)
      results_err <<- c(results_err, geterrmessage())
    }
  )
}

# Configure if should apply Box-cox transformation
transform = TRUE

##################################
# Import Time Series Data
##################################

data <- read_csv("data.csv", col_types = cols())
data$GRPRatingsDate = as.Date(data$GRPRatingsDate)
data$t = 1:nrow(data)

train_indexes = 1:72
test_indexes = 73:92

str(data)
dim(data)
summary(data)
head(data)

data_train = data$GRP[train_indexes]
data_test = data$GRP[test_indexes]

if(transform){
  lambda = BoxCox.lambda(data_train)
  data_train = BoxCox(data_train, lambda)
}

# Frequency = 52
tsdata52_train = ts(data_train, frequency=52, start=c(1, 25))
tsdata52_test = ts(data_test, frequency=52, start=c(2, 45))
tsdata52_train
tsdata52_test

# Frequency = 26
tsdata26_train = ts(data_train, frequency=26, start=c(1, 25))
tsdata26_test = ts(data_test, frequency=26, start=c(4, 19))
tsdata26_train
tsdata26_test

# Frequency = 13
tsdata13_train = ts(data_train, frequency=13, start=c(1, 12))
tsdata13_test = ts(data_test, frequency=13, start=c(7, 6))
tsdata13_train
tsdata13_test

##################################
# Exponential Smoothing
##################################

train_ses = function(data_train, data_test, frequency){
  tryCatch(
    {
      method = 'Exponential Smoothing'
      model = ses(data_train, h=20)
      if(transform) pred = boxcox.inv(model$mean,lambda) else pred = model$mean
      results_method <<- c(results_method, method)
      results_model <<- c(results_model, model$method)
      results_rmse <<- c(results_rmse, RMSE(pred, data_test))
      results_mape <<- c(results_mape, MAPE(pred, data_test))
      results_mad <<- c(results_mad, MAD(BoxCox(pred, lambda), data_test))
      results_freq <<- c(results_freq, frequency)
      results_pred <<- c(results_pred, paste(pred, collapse=','))
      results_err <<- c(results_err, NA)
    }
    ,error=logerrors(e, method, frequency))
}

train_ses(tsdata52_train, tsdata52_test, 52)
train_ses(tsdata26_train, tsdata26_test, 26)
train_ses(tsdata13_train, tsdata13_test, 13)

train_ets = function(data_train, data_test, frequency){
  tryCatch(
    {
      method = 'Exponential Smoothing'
      model = forecast(ets(data_train), h=20)
      if(transform) pred = boxcox.inv(model$mean,lambda) else pred = model$mean
      results_method <<- c(results_method, method)
      results_model <<- c(results_model, model$method)
      results_rmse <<- c(results_rmse, RMSE(pred, data_test))
      results_mape <<- c(results_mape, MAPE(pred, data_test))
      results_mad <<- c(results_mad, MAD(pred, data_test))
      results_freq <<- c(results_freq, frequency)
      results_pred <<- c(results_pred, paste(pred, collapse=','))
      results_err <<- c(results_err, NA)
    }
    ,error=logerrors(e, method, frequency))
}

train_ets(tsdata52_train, tsdata52_test, 52)
train_ets(tsdata26_train, tsdata26_test, 26)
train_ets(tsdata13_train, tsdata13_test, 13)

##################################
# ARIMA Model
##################################

train_arima = function(data_train, data_test, frequency){
  tryCatch(
    {
      method = 'ARIMA'
      model = forecast(auto.arima(data_train), h=20)
      if(transform) pred = boxcox.inv(model$mean,lambda) else pred = model$mean
      results_method <<- c(results_method, method)
      results_model <<- c(results_model, model$method)
      results_rmse <<- c(results_rmse, RMSE(pred, data_test))
      results_mape <<- c(results_mape, MAPE(pred, data_test))
      results_mad <<- c(results_mad, MAD(pred, data_test))
      results_freq <<- c(results_freq, frequency)
      results_pred <<- c(results_pred, paste(pred, collapse=','))
      results_err <<- c(results_err, NA)
    }
    ,error=logerrors(e, method, frequency))
}

train_arima(tsdata52_train, tsdata52_test, 52)
train_arima(tsdata26_train, tsdata26_test, 26)
train_arima(tsdata13_train, tsdata13_test, 13)

############################################################################################
# Decomposition Model
# The first letter denotes the error type ("A", "M" or "Z"); 
# The second letter denotes the trend type ("N","A","M" or "Z")
# The third letter denotes the season type ("N","A","M" or "Z")
# In all cases, "N"=none, "A"=additive, "M"=multiplicative and "Z"=automatically selected.
############################################################################################

train_stl_arima = function(data_train, data_test, frequency){
  tryCatch(
    {
      method = 'Decomposition'
      model = stlf(data_train, method="arima", h=20)
      if(transform) pred = boxcox.inv(model$mean,lambda) else pred = model$mean
      results_method <<- c(results_method, method)
      results_model <<- c(results_model, model$method)
      results_rmse <<- c(results_rmse, RMSE(pred, data_test))
      results_mape <<- c(results_mape, MAPE(pred, data_test))
      results_mad <<- c(results_mad, MAD(pred, data_test))
      results_freq <<- c(results_freq, frequency)
      results_pred <<- c(results_pred, paste(pred, collapse=','))
      results_err <<- c(results_err, NA)
    }
    ,error=logerrors(e, method, frequency))
}

train_stl_arima(tsdata52_train, tsdata52_test, 52)
train_stl_arima(tsdata26_train, tsdata26_test, 26)
train_stl_arima(tsdata13_train, tsdata13_test, 13)

train_stl_ets = function(data_train, data_test, frequency){
  tryCatch(
    {
      method = 'Decomposition'
      model = stlf(data_train, method="ets", h=20)
      if(transform) if(transform) pred = boxcox.inv(model$mean,lambda) else pred = model$mean else pred = boxcox.inv(model$mean,lambda)
      results_method <<- c(results_method, method)
      results_model <<- c(results_model, model$method)
      results_rmse <<- c(results_rmse, RMSE(pred, data_test))
      results_mape <<- c(results_mape, MAPE(pred, data_test))
      results_mad <<- c(results_mad, MAD(pred, data_test))
      results_freq <<- c(results_freq, frequency)
      results_pred <<- c(results_pred, paste(pred, collapse=','))
      results_err <<- c(results_err, NA)
    }
    ,error=logerrors(e, method, frequency))
}

train_stl_ets(tsdata52_train, tsdata52_test, 52)
train_stl_ets(tsdata26_train, tsdata26_test, 26)
train_stl_ets(tsdata13_train, tsdata13_test, 13)

##################################
# Time Series Regression Model
##################################

lm1 = lm(GRP~t, data = data[train_indexes, ])
lm2 = lm(GRP~t+I(t^2), data = data[train_indexes, ])
lm3 = lm(GRP~t+I(t^2)+I(t^3), data = data[train_indexes, ])
lm4 = lm(GRP~t+I(t^2)+I(t^3)+I(t^4), data = data[train_indexes, ])
lm5 = lm(GRP~t+I(t^2)+I(t^3)+I(t^4)+I(t^5), data = data[train_indexes, ])

models = list(lm1, lm2, lm3, lm4, lm5)

index = 1
for(model in models){
  results_method = c(results_method, 'Time Series Regression')
  results_model = c(results_model, paste(gsub('()', '',formula(model)[2]), gsub('()', '',formula(model)[3]), sep=' ~ '))
  results_rmse = c(results_rmse, RMSE(predict(model, data)[test_indexes], data$GRP[test_indexes]))
  results_mape = c(results_mape, MAPE(predict(model, data)[test_indexes], data$GRP[test_indexes]))
  results_mad <<- c(results_mad, MAD(predict(model, data)[test_indexes], data$GRP[test_indexes]))
  results_freq = c(results_freq, NA)
  results_pred <<- c(results_pred, paste(predict(model, data)[test_indexes], collapse=','))
  results_err <<- c(results_err, NA)
  index = index + 1
}

####################
# Model Evaluation
####################

comparison = data.frame(Method=results_method,
                        Frequency=results_freq,
                        Model=results_model, 
                        RMSE=results_rmse, 
                        MAPE=results_mape,
                        MAD=results_mad,
                        Error=results_err,
                        Forecast=results_pred)
write.csv(comparison, file='ts_result.csv', row.names=F, na='')

#############################
# Plot decomposition graphs
#############################

plot(decompose(tsdata52_train, type="additive"))
plot(decompose(tsdata26_train, type="additive"))
plot(decompose(tsdata13_train, type="additive"))

plot(decompose(tsdata52_train, type="multiplicative"))
plot(decompose(tsdata26_train, type="multiplicative"))
plot(decompose(tsdata13_train, type="multiplicative"))
