#!/usr/bin/env Rscript
library(grpreg)
Xdata <- read.csv(file="sglX.dat", header=FALSE, sep=",")
ydata <- read.csv(file="sgly.dat", header=FALSE, sep=",")
groupid <- read.csv(file="sglgroup.dat", header=FALSE, sep=",")
Xm <- as.matrix(Xdata)
yv <- ydata[["V1"]]
gid <- groupid[["V1"]]
cvfit <- cv.grpreg(Xm, yv, gid, penalty="grLasso", family="binomial", alpha=0.95, lambda.min=0.0001)
weights <- cvfit[['fit']][['beta']][,cvfit[['min']]]
write.csv(weights, file="grpW.csv")
