# Plot hair density
library(ggplot2)
library(dplyr)
library(bootstrap)
rm(list = ls())

# Set a random seed
set.seed(20)

# Load in male and female hair density measures
df_f <- read.csv("hair_f_density.csv")
df_f$Gender = "Female"
df_m <- read.csv("hair_m_density.csv")
df_m$Gender = "Male"

df <- rbind(df_f, df_m)

# Just care about 1930 on
df <- subset(df, year >= 1930)

# For each year get the median hair density
df_sum <- df %>%
  group_by(Gender, year) %>%
  summarise(med = median(density))

##### Write out the data frames of smoothed hair size

# Smooth it! Loess span = 0.5. I prefer to err on the side of smoother, but this is not based on a rigorous calculation of the best span.
# Choosing a relatively wide span (stronger smoothing) draws attention away from blips that may or may not be meaningful. 
loessmod50f <- loess(med ~ year, data = f_hair, span = 0.5)
loessmod50m <- loess(med ~ year, data = m_hair, span = 0.5)

f_hair$smoothed <- predict(loessmod50f)
m_hair$smoothed <- predict(loessmod50m)

write.csv(f_hair, "trend_f_smoothed.csv", row.names = FALSE)
write.csv(m_hair, "trend_m_smoothed.csv", row.names = FALSE)


# Make a plot for sanity
allhair <- rbind(f_hair, m_hair)
ggplot(allhair, aes(year, smoothed))+
  geom_line(aes(group = Gender, colour = Gender))

