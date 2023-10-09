#hiearchical clustering routine in r
library(stylo)
library(ggplot2)
library(BBmisc)
library(lsa)
library(stringr)

#load data, first combined datasets
MCMC_combined <- read.csv("MCMC_combined_cluster.csv",header=FALSE)
SDSS_noSize_combined <- read.csv("SDSS_noSize_combined_cluster.csv",header=FALSE)
SDSS_wSize_combined <- read.csv("SDSS_wSize_combined_cluster.csv",header=FALSE)
#then individual
# MCMC_singlePeak <- read.csv("MCMC_singlePeak_cluster.csv",header=FALSE)
# MCMC_doublePeak <- read.csv("MCMC_doublePeak_cluster.csv",header=FALSE)
# SDSS_noSize_STAR <- read.csv("SDSS_noSize_STAR_cluster.csv",header=FALSE)
# SDSS_noSize_GALAXY <- read.csv("SDSS_noSize_GALAXY_cluster.csv",header=FALSE)
# SDSS_noSize_QSO <- read.csv("SDSS_noSize_QSO_cluster.csv",header=FALSE)
# SDSS_wSize_STAR <- read.csv("SDSS_wSize_STAR_cluster.csv",header=FALSE)
# SDSS_wSize_GALAXY <- read.csv("SDSS_wSize_GALAXY_cluster.csv",header=FALSE)
# SDSS_wSize_QSO <- read.csv("SDSS_wSize_QSO_cluster.csv",header=FALSE)

#sample n rows for memory/performance
n <- 10000
MCMC_combined <- MCMC_combined[sample(nrow(MCMC_combined), n), ]
SDSS_noSize_combined <- SDSS_noSize_combined[sample(nrow(SDSS_noSize_combined), n), ]
SDSS_wSize_combined <- SDSS_wSize_combined[sample(nrow(SDSS_wSize_combined), n), ]

# MCMC_singlePeak <- MCMC_singlePeak[sample(nrow(MCMC_singlePeak), n), ]
# MCMC_doublePeak <- MCMC_doublePeak[sample(nrow(MCMC_doublePeak), n), ]
# SDSS_noSize_STAR <- SDSS_noSize_STAR[sample(nrow(SDSS_noSize_STAR), n), ]
# SDSS_noSize_GALAXY <- SDSS_noSize_GALAXY[sample(nrow(SDSS_noSize_GALAXY), n), ]
# SDSS_noSize_QSO <- SDSS_noSize_QSO[sample(nrow(SDSS_noSize_QSO), n), ]
# SDSS_wSize_STAR <- SDSS_wSize_STAR[sample(nrow(SDSS_wSize_STAR), n), ]
# SDSS_wSize_GALAXY <- SDSS_wSize_GALAXY[sample(nrow(SDSS_wSize_GALAXY), n), ]
# SDSS_wSize_QSO <- SDSS_wSize_QSO[sample(nrow(SDSS_wSize_QSO), n), ]

#normalize
MCMC_combined <- normalize(MCMC_combined)
SDSS_noSize_combined <- normalize(SDSS_noSize_combined)
SDSS_wSize_combined <- normalize(SDSS_wSize_combined)

# MCMC_singlePeak <- normalize(MCMC_singlePeak)
# MCMC_doublePeak <- normalize(MCMC_doublePeak)
# SDSS_noSize_STAR <- normalize(SDSS_noSize_STAR)
# SDSS_noSize_GALAXY <- normalize(SDSS_noSize_GALAXY)
# SDSS_noSize_QSO <- normalize(SDSS_noSize_QSO)
# SDSS_wSize_STAR <- normalize(SDSS_wSize_STAR)
# SDSS_wSize_GALAXY <- normalize(SDSS_wSize_GALAXY)
# SDSS_wSize_QSO <- normalize(SDSS_wSize_QSO)

#calculate dissimilarity matrices
d_MCMC_combined <- as.dist(1 - cosine(as.matrix(MCMC_combined)))
d_SDSS_noSize_combined <- as.dist(1 - cosine(as.matrix(SDSS_noSize_combined)))
d_SDSS_wSize_combined <- as.dist(1 - cosine(as.matrix(SDSS_wSize_combined)))

# d_MCMC_singlePeak <- as.dist(1 - cosine(as.matrix(MCMC_singlePeak)))
# d_MCMC_doublePeak <- as.dist(1 - cosine(as.matrix(MCMC_doublePeak)))
# d_SDSS_noSize_STAR <- as.dist(1 - cosine(as.matrix(SDSS_noSize_STAR)))
# d_SDSS_noSize_GALAXY <- as.dist(1 - cosine(as.matrix(SDSS_noSize_GALAXY)))
# d_SDSS_noSize_QSO <- as.dist(1 - cosine(as.matrix(SDSS_noSize_QSO)))
# d_SDSS_wSize_STAR <- as.dist(1 - cosine(as.matrix(SDSS_wSize_STAR)))
# d_SDSS_wSize_GALAXY <- as.dist(1 - cosine(as.matrix(SDSS_wSize_GALAXY)))
# d_SDSS_wSize_QSO <- as.dist(1 - cosine(as.matrix(SDSS_wSize_QSO)))

#get clustering results
method <- "ward.D" 
MCMC_combined_hclust <- hclust(d_MCMC_combined, method = method, members = NULL)
SDSS_noSize_combined_hclust <- hclust(d_SDSS_noSize_combined, method = method, members = NULL)
SDSS_wSize_combined_hclust <- hclust(d_SDSS_wSize_combined, method = method, members = NULL)

# MCMC_singlePeak_hclust <- hclust(d_MCMC_singlePeak, method = method, members = NULL)
# MCMC_doublePeak_hclust <- hclust(d_MCMC_doublePeak, method = method, members = NULL)
# SDSS_noSize_STAR_hclust <- hclust(d_SDSS_noSize_STAR, method = method, members = NULL)
# SDSS_noSize_GALAXY_hclust <- hclust(d_SDSS_noSize_GALAXY, method = method, members = NULL)
# SDSS_noSize_QSO_hclust <- hclust(d_SDSS_noSize_QSO, method = method, members = NULL)
# SDSS_wSize_STAR_hclust <- hclust(d_SDSS_wSize_STAR, method = method, members = NULL)
# SDSS_wSize_GALAXY_hclust <- hclust(d_SDSS_wSize_GALAXY, method = method, members = NULL)
# SDSS_wSize_QSO_hclust <- hclust(d_SDSS_wSize_QSO, method = method, members = NULL)

#plot dendrograms
MCMC_labels <- c("i","r̄","Mfac","rFac","f1","f2","f3","f4","Pa","Sα","PhaseAmplitude","tDelay","RBLR","FHWM")
SDSS_noSize_labels <- c("redshift","spectro_i","spectro_z","spectro_u","spectro_g","ra","dec","photo_u","photo_g","photo_r","photo_i","photo_z")
SDSS_wSize_labels <- c("redshift","spectro_i","spectro_z","spectro_u","spectro_g","ra","dec","photo_u","photo_g","photo_r","photo_i","photo_z","rad_u","rad_g","rad_r","rad_i","rad_z")

png("MCMC_combined_dendrogram.png")
plot(MCMC_combined_hclust, labels = MCMC_labels, main = str_interp("Disk-wind model dendrogram, N = ${n}"))
dev.off()

png("SDSS_noSize_combined_dendrogram.png")
plot(SDSS_noSize_combined_hclust, labels = SDSS_noSize_labels, main = str_interp("SDSS (no sizes) dendrogram, N = ${n}"))
dev.off()

png("SDSS_wSize_combined_dendrogram.png")
plot(SDSS_wSize_combined_hclust, labels = SDSS_wSize_labels, main = str_interp("SDSS (with sizes) dendrogram, N = ${n}"))
dev.off()