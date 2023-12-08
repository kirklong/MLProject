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
library(arulesViz)

# Load data
data <- read.csv('FinalExamFakeData.csv',colClasses=c("NULL",NA,NA,NA))#[,c('Top item (excluding searched item) bought by users with similar search','Most recent search','Most recent purchase')]
search <- data[,2]
often_bought <- data[,1]
purchase <- data[,3]
df <- data.frame(search,purchase,often_bought)
#use apriori to generate rules
support_threshold <- 0.1
confidence_threshold <- 0.2
minlen <- 2

rules <- apriori(df, parameter = list(supp = support_threshold, conf = confidence_threshold, minlen = minlen))