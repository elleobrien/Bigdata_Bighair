rm(list = ls())
library(reshape2)
library(dplyr)

setwd("P://Image//Bigdata_Bighair/Mullets_n_beehives")

# Count the mullets!
mullist = list.files("./Mullet_samples")
beelist = list.files("./Beehive_samples")
strlist = list.files("./Straight_samples")

# Get big directory of all pics in the dataset
df <- read.csv("../Gender_trends/coordinates.csv")

# Get the year and state
split_out <- colsplit(df$file, "_", names = c("year","rest"))
split_out2 <- colsplit(split_out$rest, "_", names = c("state","rest"))
df$year <- split_out$year
df$state <- split_out2$state

# We only care about post-1930 so trim the dataset
df <- df %>%
  subset(year >= 1930)

# Is it a file in the mullet hair list?
df$file <- gsub("-hair.png", ".png",df$file)
df$is_mullet <- ifelse(df$file %in% mullist,1,0)
df$is_beehive <- ifelse(df$file %in% beelist,1,0)
df$is_straight <- ifelse(df$file %in% strlist,1,0)

# Get the number of mullets by year
yearly <- df %>%
  group_by(year)%>%
  summarise(mullet_density = mean(is_mullet),
            beehive_density = mean(is_beehive),
            str_density = mean(is_straight))

ggplot(yearly, aes(year, str_density))+
  geom_point()


write.csv(yearly,"Hairdos_by_year.csv", row.names = FALSE)

# Look at regional differences
# Let's create some regional mappings, according to the us census bureau
ne <- c("Maine","New-Hampshire","Vermont","Massachusetts","Rhode-Island","Connecticut","New-York","New-Jersey","Pennsylvania")
mw <- c("Ohio","Michigan","Indiana","Wisconsin","Illinois","Minnesota","Iowa","Missouri","North-Dakota","South-Dakota","Nebraska","Kansas")
so <- c("Delware","Maryland","Virginia","West-Virginia","Kentucky","North-Carolina","South-Carolina","Tennessee","Georgia","Florida","Alabama",
        "Mississippi","Arkansas","Louisiana","Texas","Oklahoma")
we <- c("Montana","Idaho","Wyoming","Colorado","New-Mexico","Arizona","Utah","Nevada","California","Oregon","Washington","Alaska","Hawaii")

df$region <- ifelse(df$state %in% ne, "Northeast",
                    ifelse(df$state %in% mw, "Midwest",
                           ifelse(df$state %in% so, "South",
                                  ifelse(df$state %in% we, "West",
                                         "Not found"))))

# Look at regional popularity
geographic <- df %>%
  group_by(region)%>%
  summarise(mullet_density = mean(is_mullet),
            beehive_density = mean(is_beehive),
            str_density = mean(is_straight))
write.csv(geographic,"Hairdos_by_region.csv", row.names = FALSE)


#geographic <- df %>%
#  group_by(region,year)%>%
#  summarise(mullet_density = mean(is_mullet),
#            beehive_density = mean(is_beehive))

#ggplot(geographic, aes(year,mullet_density))+
#  geom_point(aes(group = region, colour = region))

