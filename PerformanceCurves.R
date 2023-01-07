## Plots for Data Driven Performance Evaluation
## plot the ROC curve for classification of y with p
##
roc <- function(p,y, ...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  Q <- p > matrix(rep(seq(0,1,length=100),n),ncol=100,byrow=TRUE)
  specificity <- colMeans(!Q[y==levels(y)[1],])
  sensitivity <- colMeans(Q[y==levels(y)[2],])
  plot(1-specificity, sensitivity,  ylab="TP", xlab="FPR",type="l", main="ROC Curve", ...)
  abline(a=0,b=1,lty=2,col=8)
  ROCcurve <-as.data.frame( cbind( 1-specificity,  sensitivity))
  return (ROCcurve)
}


profitcurveAll <- function(p,y, cost.benefit.m,...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  pp <- p[order(p, decreasing =TRUE, na.last = NA)]
  yy <- y[order(p, decreasing =TRUE, na.last = NA)]
  Q <- pp > matrix(rep(seq(0,1,length=100),n),ncol=100,byrow=TRUE)
  TN <- colSums(!Q[yy==levels(yy)[1],])
  FN <- colSums(Q[yy==levels(yy)[1],])
  TP <- colSums(Q[yy==levels(yy)[2],])
  FP <- colSums(!Q[yy==levels(yy)[2],])
  profit <- cost.benefit.m %*% rbind( TP, FP, FN, TN ) 
  plot(seq(0,1,length=100), profit, type="l", xlab="Proportion of data", ...)
 
}

profitcurve <- function(p,y, cost.benefit.m ,...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  pp <- p[order(p, decreasing =TRUE, na.last = NA)]
  yy <- y[order(p, decreasing =TRUE, na.last = NA)]
  profit <- rep.int(0,n)
  VV1<-(yy==levels(yy)[1])
  VV2<-(yy==levels(yy)[2])
  for ( kk in 1:n ){
    TN <- 0
    FN <- 0
    TP <- sum(VV2[1:kk])
    FP <- sum(!VV2[1:kk])
    profit[kk] <- cost.benefit.m %*% rbind( TP, FP, FN, TN ) 
  }
  plot(seq(from=1,to=n,by=1)/n, profit, type="l", xlab="Proportion of data", ylab="Profit", main="Profit Curve", ...)
  return (profit)
}

cumulativecurve <- function(p,y,...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  pp <- p[order(p, decreasing =TRUE, na.last = NA)]
  yy <- y[order(p, decreasing =TRUE, na.last = NA)]
  cumulative <- rep.int(0,n)
  VV1<-(yy==levels(yy)[1])
  VV2<-(yy==levels(yy)[2])
  for ( kk in 1:n ){
    TP <- sum(VV2[1:kk])
    ## True positive / total number of positive
    cumulative[kk] <- TP/sum(VV2)
  }
plot(seq(from=1,to=n,by=1)/n, cumulative, type="l", xlab="Proportion of data targeted", ylab="Proportion of Total Positive", main="Cumulative Response Curve", ...)
return (cumulative)
}
liftcurve <- function(p,y,...){
  y <- factor(y)
  n <- length(p)
  p <- as.vector(p)
  pp <- p[order(p, decreasing =TRUE, na.last = NA)]
  yy <- y[order(p, decreasing =TRUE, na.last = NA)]
  lift <- rep.int(0,n)
  VV1<-(yy==levels(yy)[1])
  VV2<-(yy==levels(yy)[2])
  for ( kk in 1:n ){
    TP <- sum(VV2[1:kk])
    lift[kk] <- ( TP/sum(VV2) ) / ( kk/n ) 
  }
  plot(seq(from=1,to=n,by=1)/n, lift, type="l", main="Lift Curve", xlab="Proportion of data targeted", ylab="Lift", ...)
  return (lift)
}

