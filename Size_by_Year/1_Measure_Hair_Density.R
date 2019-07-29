# Measure hair densities for women's hair
library(png)
library(reshape2)
rm(list = ls())

gender = "Male"

if (gender == "Female"){
  mydir = "../Hair_Maps/F_hair"
}else{
  mydir = "../Hair_Maps/M_hair"
}

files = list.files(mydir)

df <- data.frame(filename = files)
split_out <- colsplit(df$filename, "_", names = c("year","rest"))
split_out2 <- colsplit(split_out$rest, "_", names = c("state","rest"))
df$year <- split_out$year
df$state <- split_out2$state


hair_density = vector()
for (i in seq(1,nrow(df))){
  path = file.path(mydir,files[i])
  img <- readPNG(path)
  hair_density[i] <- sum(1 - img)/(256*256)
  

}
df$density <- hair_density

if (gender == "Female"){
  fid = "hair_f_density.csv"
}else{
  fid = "hair_m_density.csv"
}

write.csv(df,fid, row.names = FALSE)

