########################################################
### Get the hair size for each of the sample images ####
########################################################
library(dplyr)
library(stringr)
library(ggplot2)
library(cowplot)
rm(list = ls())

# Load in some lookup tables
f_size <- read.csv("hair_f_density.csv")
m_size <- read.csv("hair_m_density.csv")
size_df <- rbind(f_size, m_size)

# A little formatting so that the filenames/dataframe columns are compatible with the format of the .csv of image samples
names(size_df)[1] <- "file"
size_df$file = str_replace(size_df$file, "-hair.png",".png")


# Load in a list of our sample images
samples <- read.csv("../Looks_by_Decade/Most_representative_images_by_FiveYear.csv")

# Merge these dataframes!
df_sum <- merge(size_df, samples, by = "file")

write.csv(df_sum, "Sample_image_hair_size_fiveyear.csv", row.names = FALSE)



####################
#### A little sanity checking plot ####
###################
samp <- df_sum[sample(nrow(df_sum), 6), ]
samp <- samp[order(samp$density),]
get_list = samp$file

img_files = file.path("../Looks_by_Decade/Representative_of_FiveYear", get_list)

p1 <- ggdraw() +
  draw_image(img_files[1], scale = 1)
p2 <- ggdraw() +
  draw_image(img_files[2], scale = 1)
p3 <- ggdraw() +
  draw_image(img_files[3], scale = 1)
p4 <- ggdraw() +
  draw_image(img_files[4], scale = 1)
p5 <- ggdraw() +
  draw_image(img_files[5], scale = 1)
p6 <- ggdraw() +
  draw_image(img_files[6], scale = 1)

labels <- round(samp$density,2)

px <- plot_grid(p1,p2,p3,
                p4,p5,p6,
                labels = labels,
                nrow = 2)
px
