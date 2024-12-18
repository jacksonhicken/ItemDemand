
library(vroom)
library(tidyverse)
library(gridExtra)
library(patchwork)
library(parsnip)
library(modeltime)
library(timetk)
library(tidymodels)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(dbarts)
library(embed)
library(naivebayes)
library(discrim)

# loading data ------------------------------------------------------------

item_train <- vroom("train.csv")
item_test <- vroom("test.csv")

item1store1 <- item_train %>% 
  filter(store == 1, item == 1)

item1store1_test <- item_test %>% 
  filter(store == 1, item == 1)

item2store2 <- item_train %>% 
  filter(store == 2, item == 2)

item2store2_test <- item_test %>% 
  filter(store == 2, item == 2)


# EDA ---------------------------------------------------------------------

ts11 <- item1store1 %>%
  ggplot(mapping=aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se=FALSE) +
  labs(title = "Store 1 Item 1")

ts22 <- item2store2 %>%
  ggplot(mapping=aes(x = date, y = sales)) +
  geom_line() +
  geom_smooth(se=FALSE) +
  labs(title = "Store 2 Item 2")

acfm11 <- item1store1 %>%
  pull(sales) %>% 
  forecast::ggAcf(.)

acfm22 <- item2store2 %>%
  pull(sales) %>% 
  forecast::ggAcf(.)

acfy11 <- item1store1 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

acfy22 <- item2store2 %>%
  pull(sales) %>% 
  forecast::ggAcf(., lag.max=2*365)

(ts11 + ts22) / (acfm11 + acfm22) / (acfy11 + acfy22)


# random forest -----------------------------------------------------------

my_recipe <- recipe(sales~., data = item1store1) %>% 
  step_date(date, features="dow") %>% 
  step_date(date, features="month") %>% 
  step_date(date, features="year")

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model)

tuning_grid <- grid_regular(mtry(range= c(1,length(item1store1)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(item1store1, v=5, repeats = 1)

CV_results <- rf_wf %>% 
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

bestTune <- CV_results %>% 
  select_best()

final_wf <-rf_wf  %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = item1store1)

rf_preds <-final_wf %>% 
  predict(new_data = , type = "regression")

collect_metrics(CV_results)

kaggle_submission <-tibble(id =ggg_test$id,
                           type = rf_preds$.pred_class)

vroom_write(x=kaggle_submission, file="./ggg_rf4.csv", delim=",")


# ARIMA -------------------------------------------------------------------

cv_split <- time_series_split(item1store1, assess="3 months", cumulative = TRUE)

cv_split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales~., data=item1store1) %>%
  step_rm(item, store) %>%
  step_date(date, features=c("doy", "decimal")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_rm(date_doy)

arima_model <- arima_reg() %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))
## Visualize results
p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = training(cv_split)
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

## Refit to whole data
fullfit <- cv_results %>%
  modeltime_refit(data = item1store1)

p2 <- fullfit %>%
  modeltime_forecast(
    new_data = item1store1_test ,
    actual_data = item1store1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

cv_split <- time_series_split(item2store2, assess="3 months", cumulative = TRUE)

cv_split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales~., data=item2store2) %>%
  step_rm(item, store) %>%
  step_date(date, features=c("doy", "decimal")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_rm(date_doy)

arima_model <- arima_reg() %>%
  set_engine("auto_arima")

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))
## Visualize results
p3 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = training(cv_split)
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

## Refit to whole data
fullfit <- cv_results %>%
  modeltime_refit(data = item1store1)

p4 <- fullfit %>%
  modeltime_forecast(
    new_data = item1store1_test ,
    actual_data = item1store1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plotly::subplot(p1,p3,p2,p4, nrows=2)

# facebook prophet --------------------------------------------------------

cv_split <- time_series_split(item1store1, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split))

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))

p5 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = training(cv_split)
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

fullfit <- cv_results %>%
  modeltime_refit(data = item1store1)

p6 <- fullfit %>%
  modeltime_forecast(
    new_data = item1store1_test ,
    actual_data = item1store1
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

cv_split <- time_series_split(item2store2, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split))

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))

p7 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = training(cv_split)
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

fullfit <- cv_results %>%
  modeltime_refit(data = item2store2)

p8 <- fullfit %>%
  modeltime_forecast(
    new_data = item2store2_test ,
    actual_data = item2store2
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

plotly::subplot(p5,p7,p6,p8, nrows=2)
