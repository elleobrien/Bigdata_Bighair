rm(list = ls())
library(ggplot2)
library(dplyr)
library(reshape2)
library(MASS)
library(bootstrap)

df <- read.csv("coordinates.csv")
head(df)
df$is_female <- as.factor(df$is_female)
df$Gender <- ifelse(df$is_female == 1, "Female",
                    "Male")

# Get year and state
split_out <- colsplit(df$file, "_", names = c("year","rest"))
split_out2 <- colsplit(split_out$rest, "_", names = c("state","rest"))
df$year <- split_out$year
df$state <- split_out2$state

df <- subset(df, year >= 1930)
years <- unique(df$year)
# How confusible are the two genders?
discrim <- vector()
i = 1

for (d in years){
  sub <- df %>% subset(year == d)
  
  # Train test split (80-20%)
  #train.index <- createDataPartition(sub$Gender, p = 0.8, list = FALSE)
  
  #train <- sub[train.index,]
  #test <- sub[-train.index,]
  
  #qda_fit <- qda(is_female ~ f1 + f2 + f3 + f4, train)
  #pred <- predict(qda_fit, test)
  #accuracy <- mean(pred$class == test$is_female)
  ctrl <- trainControl(method = "cv",
                       p = 1,
                       number = 5
                       )
  
  mod <- train(is_female ~ f1+f2+f3+f4, 
               data = sub,
               method = "rf",
               trControl = ctrl)
  
  
  discrim[i] <- max(mod$results$Accuracy)
  i = i+ 1
}


discrim_df <- data.frame(discrim = discrim,
                         year = years)

# Take a look at the results
px <- ggplot(discrim_df, aes(year, discrim))+
  geom_point()+
  geom_smooth(se = FALSE, method = "loess", size = 2)+
  xlab("Decade")+
  ylab("Male-female discriminability")+
  scale_x_continuous(breaks = seq(1930,2010, by = 10))+
  theme_bw()+
  scale_y_continuous(labels = scales::percent_format(accuracy = 1))+
  theme(axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

px



###############################################################################
################## SMOOTHING ##################################################
# Identify the ideal span of the loess smoother using cross-validation #######
loessmod50 <- loess(discrim ~ year, data = discrim_df, span = 0.5)
discrim_df$smooth <- predict(loessmod50)

write.csv(discrim_df, "Discriminability_by_year.csv", row.names = FALSE)


