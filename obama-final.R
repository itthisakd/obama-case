

# 1: Data Preparation
install.packages(c("glmnet", "usmap", "rpart", "Metrics", "randomForest", "caret", "mlbench", "ellipse" ))
library(randomForest)
library(tidyverse)
library(dplyr)
library(Metrics)
library(glmnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(mlbench)
library(caret)
set.seed(999)

elect.df <- read.csv('~/Desktop/ucl/25DA2/Obama.csv')

nrow(elect.df)

elect.df.known %>% count(ElectionDate)

names(elect.df)
startCol <- which(names(elect.df) == "MalesPer100Females")
endCol <- which(names(elect.df) == "FarmArea")

# deriving target attribute
elect.df$ObamaPercentMargin <-
  100*(elect.df$Obama - elect.df$Clinton) / elect.df$TotalVote

# --------- Data Cleaning, fill in NAs ----------

countNAs <- function (v) sum(is.na(v))
elect.countNAs <- sapply(elect.df, countNAs)
elect.countNAs[elect.countNAs != 0]

sapply(elect.df[,c('AverageIncome', 'MedianIncome')], countNAs)

cor(x = elect.df$AverageIncome, y = elect.df$MedianIncome, use = "complete.obs")

ggplot(elect.df, 
       aes(x = AverageIncome, 
           y = MedianIncome)) + 
  geom_point() + 
  geom_abline(intercept = 0, 
              slope = 1)

elect.df$AverageIncome <- ifelse(is.na(elect.df$AverageIncome), 
                                 elect.df$MedianIncome, 
                                 elect.df$AverageIncome)
summary(elect.df$AverageIncome)

# replace NAs by national averages via a loop over attribute names
for ( attr in c("Black", "Asian", "AmericanIndian", "ManfEmploy", "Disabilities", 
                "DisabilitiesRate", "FarmArea") ){ 
  elect.df[[attr]][is.na(elect.df[[attr]])] <- mean(elect.df[[attr]], na.rm = TRUE)
}

elect.countNAs <- sapply(elect.df, countNAs)
elect.countNAs[elect.countNAs != 0]

na.idx <- seq(1:nrow(elect.df))[!complete.cases(elect.df[,startCol:endCol])]
elect.df <- elect.df[-na.idx,]

elect.countNAs <- sapply(elect.df, countNAs)
elect.countNAs[elect.countNAs != 0]

# -------------------Spliting Dataset ------------------------------

elect.df.known <- elect.df[!is.na(elect.df$ObamaPercentMargin), ]

# find the total number of records in the dataset
nKnown <- nrow(elect.df.known)
rowIndicesTrain <- sample(1:nKnown, size = round(nKnown*0.75), replace = FALSE)

# create training and testing datasets
elect.df.training <- elect.df.known[rowIndicesTrain, ] # Training dataset
elect.df.test <- elect.df.known[-rowIndicesTrain, ] # Testing dataset

columns = c(6, startCol:endCol)

# training dataset
xtraining <- as.matrix(elect.df.training[, columns])
ytraining <- elect.df.training$ObamaPercentMargin
# testing dataset
xtest <- as.matrix(elect.df.test[, columns])
ytest <- elect.df.test$ObamaPercentMargin

elect.df.known %>% count(ElectionType)


# 2: Data Visualisation
# ---------------------- Visualisations ----------------


cor.ObamaPercentMargin <- cor(elect.df[,startCol:endCol], elect.df$ObamaPercentMargin,
                              use = "complete.obs")
cor.ObamaPercentMargin.df <- data.frame(cor = cor.ObamaPercentMargin, abs.cor = abs(cor.ObamaPercentMargin),
                                        row.names = rownames(cor.ObamaPercentMargin))
cor.ObamaPercentMargin.df <- cor.ObamaPercentMargin.df[order(-cor.ObamaPercentMargin.df$abs.cor),]
top10 = rownames(cor.ObamaPercentMargin.df[1:10,])

# Figure 2: Correlation between attributes in focus
cor(elect.df[,startCol:(startCol+5)],use = "complete.obs")
cor.info <- cor(elect.df[,top10], use = "complete.obs")
cor.info

library(ellipse)
library(RColorBrewer)
my_colors <- 
  colorRampPalette(brewer.pal(5, 
                              "Spectral"))(100) 
# plot the correlogram
plotcorr(cor.info, 
         col = my_colors[cor.info*50+50], 
         mar = c(0,0,0,0), 
         cex.lab = 0.7, 
         type = "upper", 
         diag = F)


# -------------------- Visualisations -------------------------

# Figure 1: Correlation between attributes and ObamaPercentMargin
ggplot(cor.ObamaPercentMargin.df,
       aes(x = reorder(row.names(cor.ObamaPercentMargin.df), -abs.cor),
           y = cor, fill = cor)) +
  geom_col() +
  ggtitle("ObamaPercentMargin: Top Postive/Negative Correlating Attributes") +
  xlab("Attribute") +
  scale_fill_gradient(low = "red", high = "green") +
  theme(axis.text.x = element_text(angle = -90, hjust = 0))

# Figure 3: Map of ObamaMargin
elect.df.known <- elect.df[!is.na(elect.df$ObamaPercentMargin), ]
elect.df.known$ObamaPercentMargin <- predict(rf, elect.df.unknown)
obamamargin <-
  data.frame(fips = elect.df.known$FIPS,
             value = elect.df.known$ObamaPercentMargin)

obamamargin

plot_usmap(data=obamamargin, regions="states", values="value", color="darkgrey", labels=TRUE) +
  scale_fill_gradient(name = "ObamaPercentMargin", low = "red", high = "green") +
  theme(panel.background = element_rect(color = "black", fill = "lightblue"),
        legend.position = "right")+labs(title="Map of Observed ObamaPercentMargin")


# Figure 4: Box plot of ObamaPercentMargin by Election Type
elect.df %>% ggplot(aes(x=ElectionType, y=ObamaPercentMargin)) + geom_boxplot() + 
  labs(title="Box plot of ObamaPercentMargin for each Election Type")

# 3: Model Creation

# ----------------------Utility Functions ------------------------------

# function to get MAE and RMSE
genError <- function(prediction, actual){
  return( 
    list(
      MAE = signif(mae(actual, prediction), 4),
      RMSE = signif(rmse(actual, prediction), 4),
      MAPE = signif(mape(actual, prediction), 4)
    )
  )
}

# function to append error to error df
appendError <- function(results, model){
  prediction = predict(model, elect.df.test)
  name = deparse(substitute(model))
  error = genError(prediction, elect.df.test$ObamaPercentMargin)
  
  return (rbind(results, data.frame(MAE = error$MAE, RMSE = error$RMSE, Model = name)))
}

#-------------------- Set up for models ------------------------

# equations for model all and insight
eqn = ObamaPercentMargin ~ ElectionType+MalesPer100Females+AgeBelow35+Age35to65+Age65andAbove+White+Black+Asian+AmericanIndian+
  Hawaiian+Hispanic+HighSchool+Bachelors+Poverty+IncomeAbove75K+MedianIncome+AverageIncome+UnemployRate+
  ManfEmploy+SpeakingNonEnglish+Medicare+MedicareRate+SocialSecurity+SocialSecurityRate+RetiredWorkers+
  Disabilities+DisabilitiesRate+Homeowner+SameHouse1995and2000+PopDensity+LandArea

# -----------------------Linear Model --------------------------

lm.all <- lm(eqn, data = elect.df.training)


lm.step.backward <- step(lm.all, direction = "backward")

# create a minimal linear regression model with no attributes to initiate the forward stepwise model selection
lm.min <- lm(ObamaPercentMargin ~ 1, data = elect.df.training)
# start the forward stepwise model selection
lm.step.forward <- step(lm.min, direction = 'forward',
                        scope = eqn)

# -----------------------Regression Tree  --------------------------

rt.all <-
  rpart(eqn,
        data = elect.df.training,
        cp = 0.001) 

plotcp(rt.all, upper = "splits")

cp.min.fct <- function(rt.model){
  df <- as.data.frame(rt.model$cptable)
  min_idx <- which.min(df$xerror)
  df$CP[min_idx]
}

cp.min <- cp.min.fct(rt.all)
print(cp.min)
rt.all.min <- prune(rt.all, cp = cp.min)

# rpart.plot(rt.all.min, type = 1, extra = 1)

# --------------------- Random Forest --------------------------------

ntree = 500
# Figure 8: best mtry
bestmtry <- tuneRF(xtraining, ytraining, stepFactor=1.5, improve=1e-5, ntree=ntree)
print(bestmtry)


bestmtry=data.frame(bestmtry)
mtry_val=bestmtry[order(bestmtry$OOBError),][, 1][1]

print("Least OOB Error is with mtry=")
print(mtry_val)

rf = randomForest(eqn, data=elect.df.training, stepFactor=1.5, improve=1e-5, mtry=mtry_val, ntree=ntree, do.trace=1000, importance=TRUE)


plot(rf)
importance(rf)
# Figure 10: Feature importance
varImpPlot(rf)

# --------------------- Errors --------------------------------

errors <- data.frame(MAE = c(), RMSE = c(), Model=c())

errors <- appendError(errors, lm.all)
errors <- appendError(errors, lm.step.backward)
errors <- appendError(errors, lm.step.forward)
errors <- appendError(errors, rt.all.min)
errors <- appendError(errors, rt.all)
errors <- appendError(errors, rf)

errors[order(errors$MAE),]

errors[order(errors$RMSE),]

# --------------------- Evaluation --------------------------------

best_model = rf

outsample.df = data.frame(
  actual=ytest,
  pred=predict(best_model, elect.df.test))

insample.df = data.frame(
  actual=ytraining,
  pred=predict(best_model, elect.df.training))


# Figure 11: Outsample accuracy
outsample.df %>% 
  ggplot(aes(actual, pred)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, color="red") + geom_smooth()+
  labs(title="Outsample Prediction VS Actual")


#count
pos_pred = outsample.df %>% 
  mutate(overestimation=(pred-actual>0)) %>% 
  filter(pred>0) %>% 
  count(overestimation) %>% 
  mutate(count=n/sum(n))

neg_pred = outsample.df %>% 
  mutate(overestimation=(pred-actual>0)) %>% 
  filter(pred<0) %>% 
  count(overestimation) %>% 
  mutate(count=n/sum(n))

overestimation_count = data.frame(
  estimation=c("under", "over"),
  pos_pred=pos_pred$count,
  neg_pred=neg_pred$count
)
overestimation_count


# 4: Predictions



# --------------- Making Predictions for df.unknown ----------------

# extract part of the dataset where no voting information are available
elect.df.unknown <- elect.df[is.na(elect.df$ObamaPercentMargin), ]
# make predictions for the 'ObamaPercentMargin' and
# add the predicted values to 'elect.df.unknown'
elect.df.unknown$PredictedObamaPercentMargin <- predict(rf, elect.df.unknown)
# create a dataframe that contains the county identifiers and the
# corresponding predictions of 'ObamaPercentMargin'
pred_obamamargin <-
  data.frame(fips = elect.df.unknown$FIPS,
             value = elect.df.unknown$PredictedObamaPercentMargin)


# # --------------------- Visualisation --------------------- 


library(usmap) # load the package

# Figure 13: Predictions of ObamaPercentMargin
plot_usmap(data=pred_obamamargin, regions="states", values="value", color="darkgrey", labels=TRUE) +
  scale_fill_gradient(name = "Pred. ObamaPercentMargin", low = "red", high = "green") +
  theme(panel.background = element_rect(color = "black", fill = "lightblue"),
        legend.position = "right")+labs(title="Map of ObamaPercentMargin Predictions")

# Figure 14: Predictions of ObamaPercentMargin where abs(ObamaPercentMargin)<5
plot_usmap(data=pred_obamamargin %>% filter(abs(value)<5), regions="states", values="value", color="darkgrey", labels=TRUE) +
  scale_fill_gradient(name = "Pred. ObamaPercentMargin", low = "orange", high = "purple") +
  theme(panel.background = element_rect(color = "black", fill = "lightblue"),
        legend.position = "right")+labs(title="Map of ObamaPercentMargin Predictions where abs(ObamaPercentMargin)<5")


# states with predicted margin less than +-5
split_counties = elect.df.unknown %>% 
  filter(abs(PredictedObamaPercentMargin)<5) %>% 
  select(State, County, PredictedObamaPercentMargin) %>% 
  arrange(abs(PredictedObamaPercentMargin))

split_counties[1:20,]


