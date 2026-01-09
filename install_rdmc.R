if (!require("remotes", quietly=TRUE)) {
  install.packages("remotes", repos="https://cloud.r-project.org")
}
remotes::install_github("CenterForPeaceAndSecurityStudies/rDMC", quiet=TRUE)
library(rDMC)
cat("Available datasets:\n")
print(data(package="rDMC")$results[,c("Item","Title")])
