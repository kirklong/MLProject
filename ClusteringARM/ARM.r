# association rule mining script

library(viridis)
library(arules)
library(TSP)
library(data.table)
library(ggplot2)
library(Matrix)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)

# load datasets
MCMC <- read.transactions("MCMCbasket.csv", format = "basket", sep = ",", skip = 1)
SDSS_noSize <- read.transactions("SDSS_noSizeBasket.csv", format = "basket", sep = ",", skip = 1)
SDSS_wSize <- read.transactions("SDSS_wSizeBasket.csv", format = "basket", sep = ",", skip = 1)

# use apriori to generate rules
support_threshold <- 0.1
confidence_threshold <- 0.2
minlen <- 2

MCMC_rules <- arules::apriori(MCMC,parameter = list(support=support_threshold,confidence=confidence_threshold,minlen=minlen))
SDSS_noSize_rules <- arules::apriori(SDSS_noSize,parameter = list(support=support_threshold,confidence=confidence_threshold,minlen=minlen))
SDSS_wSize_rules <- arules::apriori(SDSS_wSize,parameter = list(support=support_threshold,confidence=confidence_threshold,minlen=minlen))
