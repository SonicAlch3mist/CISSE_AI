###########################################################
##  @author: James Guymon 
##  @date: 7 June 2019
##  @email: james@jamesguymon.com
##  @Twitter: @JamesGuymon
#
## Forward
###########################################################
#
## This document is first and foremost pedagogical,
## prepared for the 2019 CISSE Conference:
## "Hands-on Cybersecurity Artificial Intelligence"
#
## The code is also a bit verbose in places so the processes are clear
#
## the dataset was obtained from:
## https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
## and is the work of Dr. Nour Moustafa and Jill Slay
#
## from: Designing an online and reliable statistical anomaly detection framework for dealing 
## with large high-speed network traffic. Diss. 
## University of New South Wales, Canberra, Australia, 2017.
#
## For more information, please contact the authors: Dr. Nour Moustafa and Professor Jill Slay. 
## Dr. Nour is a lecturer in Cybersecurity with SEIT-UNSW Canberra, and he is 
## interested in new Cyber threat intelligence approaches and the technology of Industry 4.0. 
#
## Disclaimer: The views and opinions expressed in this presentation and the accompanying materials 
## are the presenter's and do not necessarily reflect the opinions or positions of Ascential plc or 
## its affiliates."

###########################################################

## Getting Started
###########################################################
#
## Step 1: create a new directory for this project and save this file into it
#
## Step 2: create 'data_in' and 'viz' folders inside that directory
#
## Step 3: save the .csv files from 
## https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/ 
## into data_in
#
## Step 4: adjust line 50 (below) to point to your directory.
## an easy way to do it in RStudio via the headers nav 
## is: Session -> Set Working Directory -> To Source File Location
## then copy paste the command it executes to replace line 42

setwd("~/Projects/intrusionDetection")

## this option prevents scientific notation in plots
options(scipen=999)

## this sets some graphics params at the onset
par(mfrow=c(1,1))

## the pdf() begins a pdf doc, writing all plots to the pdf
## which will prevent them displaying in RStudio
## the file is finalized with dev.off()
pdf('./viz/presentation_plots.pdf',width=18,height=10,title="CISSE 2019 - AI Trellis Visualizations")
# ###########################################################

## Keras and TensorFlow Installation Hints
###########################################################
#
# #### When you want to explore more robust ANN's
# #### you will likely want to try Tensorflow and/or Keras. 
# ## If you haven't already, install RTools here
# 
# https://cran.r-project.org/bin/windows/Rtools/
# 
# ## Next you will need to install Keras from github
# ## as it is not available (as of May 19,2019) from CRAN
# ## To install from github you need 'devtools'
# 
# install.packages('devtools')
# 
# ## Now install Keras
# #devtools::install_github("rstudio/keras") 
#
## these instructions may prove insufficient in themselves to 
## get you up and running - they are more of a push in the
## right direction.
###########################################################

## Dependencies
###########################################################
## we check your installed packages and install what you are missing
## except for keras and tensorflow, which were explained above ^^
packages <- c("data.table", "Amelia", "igraph", "corrplot", "unbalanced"
              ,"dummies", "scales", "Boruta", "deepnet", "xgboost", "randomForest"
              , "iptools", "bit64","dplyr","caret")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}

# ## commented out because they are not used here, but recommended highly
# ## Check them out!
# extended_packages <- c("mice", "bpca", "GA", "e1071","dplyr","tidyverse")
# if (length(setdiff(extended_packages, rownames(installed.packages()))) > 0) {
#   install.packages(setdiff(extended_packages, rownames(installed.packages())))  
# }

library(data.table) ## a useful library for loading in data
library(dplyr)

## Data Visualiation Aids
library(Amelia) # good viz package for missing data
library(network) # other use as well, but useful in visualizing networks
library(corrplot) # beautiful correlation matrix plots
library(igraph) # network plots -- also library(network) is useful

# ## imputation tools you should know
# library(mice) # an historically important package for null imputation; not computationally efficient
# library(bpca) # a more modern approach worth exploring

# Data Transformation
library(unbalanced) # a package of all required tools to rebalance data sets
library(dummies) # a package to transform categorical variables into numerics
library(scales) # helpful to recale data into format that can be transformed

# Feature Selection
library(Boruta) # highly recommended gem
# library(GA) # revisit when ready for an advanced technique

# # ML algorithms
# library(keras)
# library(tensorflow); install_tensorflow()
library(deepnet) # a low-learning-curve deep learning package
library(xgboost) # a great swiss-army knife algorithm that can do both regression and classification
library(randomForest) # another staple in the ML toolkit; we load it to use in the re-balancing tournament
# library(e1071) # included here for its confusionMatrix() function, but full of goodies

# Model Measurement
library(caret) ## lots of uses here

## of particular use in this intrusion dataset
library(iptools) # transforms ip addresses into a numeric form
require(bit64) # allows 64 bit integers at import

# helper functions section
'%!in%' <- function(x,y)!('%in%'(x,y))
###########################################################

## Data Import
###########################################################
## Research Topic: Practice reading in messy data
## such as crawled html.  Challenge yourself until you 
## can read in anything.

column_names<-data.table::fread("./data_in/NUSW-NB15_features.csv",stringsAsFactors=F,data.table=F)
ground_truth<-data.table::fread("./data_in/NUSW-NB15_GT.csv",stringsAsFactors=F,data.table=F)
list_events<-data.table::fread("./data_in/UNSW-NB15_LIST_EVENTS.csv",stringsAsFactors=F,data.table=F)

## Often we need to construct large data sets from smaller files
## it is often best to do this programmatically rather than one by one
counter=0
for (chunk in 1:4){ ## for the full version run
  #for (chunk in 1:1){ # chunk=1 ## for smaller machines and during exploration for faster iteration
  counter=counter+1
  tmp_df<-fread(paste("./data_in/UNSW-NB15_",chunk,".csv",sep=''),stringsAsFactors=F,header=F,data.table=F)
  
  if (counter==1){
    intrusion_df<-as.data.frame(matrix(NA,ncol=length(tmp_df),nrow=0))
    intrusion_df[,21]<-bit64::as.integer64(intrusion_df[,21])
    intrusion_df[,22]<-bit64::as.integer64(intrusion_df[,22])
  }  
  intrusion_df<-rbind(intrusion_df,tmp_df); rm(tmp_df)
}

colnames(intrusion_df)=column_names$Name

## both 'sport' and 'dsport' shoud be integers
## but came in as characters due to '-' chars
## after reviewing the problematic rows
## we can coerce to integer, which will generate nulls
## but the '-' char represented NULL anyway.
intrusion_df$sport<-as.integer(intrusion_df$sport)
intrusion_df$dsport<-as.integer(intrusion_df$dsport)

intrusion_df$attack_cat<-NULL ## this is not needed for our purposes

## Research Topic: How could we keep and use these integer64 objects?
intrusion_df$stcpb<-NULL ## we remove this integer64 type for initial simplicity
intrusion_df$dtcpb<-NULL ## we remove this integer64 type for initial simplicity
###########################################################

## Pre-Visualization
###########################################################
## Research Topic: Scour the web for pre-viz
## approaches that catch your eye for both
## insightfulness and impact
## be careful to prioritize insight over aesthetic

## processing the full data set can often be too much.  
## we randomly select a representative sample to plot
set.seed(42)
visualize_me<-sample(size=50000,x=nrow(intrusion_df)) 
visualize_me_big<-sample(size=500000,x=nrow(intrusion_df)) 
Amelia::missmap(intrusion_df[visualize_me,]) ## plot Missingness Map

## the data represent a network, so let's take a look
## properly; at this stage we do not get fancy -
## this is just a check-in in hopes it guides our approach

## quick helper function for inside dplyr aggregation
factorMode <- function(column){ 
  factor_lvl<-which.max(tabulate(column))
  mode<-levels(column)[factor_lvl]
  return(mode)
}

# this is dplyr at work.  Note the use of the %>% operator
network_viz_df<-intrusion_df[visualize_me_big,which(colnames(intrusion_df) %in% c("srcip","dstip","Label"))]
network_viz_df<-network_viz_df %>%
  group_by(srcip) %>%
  summarise(dstip=factorMode(as.factor(dstip))
            ,worst_outcome=max(Label)
            )

problem_src_ip<-network_viz_df$srcip[which(network_viz_df$worst_outcome==1)]; problem_src_ip<-problem_src_ip[order(problem_src_ip)]
problem_dst_ip<-network_viz_df$dstip[which(network_viz_df$worst_outcome==1)]; problem_dst_ip<-problem_dst_ip[order(problem_dst_ip)]

links<-network_viz_df[,1:2]
net <- graph_from_data_frame(d=links, vertices=NULL, directed=T) 
net <- simplify(net, remove.multiple = F, remove.loops = T) 

par(mfrow=c(1,2))
plot(net,main="Network Plot with IPs")
plot(net, edge.arrow.size=.4,vertex.label=NA,main="Clean Network Plot")
print(paste("problematic srcip : ",problem_src_ip,sep=''))
print(paste("problematic dstip : ",problem_dst_ip,sep=''))

## it is a common need to only perform some processes on numerics
nums<-vector(); nums_x<-vector()

## we will need to refresh the following vars (below) from time to time
## so we make it a function to avoid copy/paste
reset_numeric_vector<-function(df){
  nums<<-which(unlist(lapply(df, is.numeric))) 
  nums_x<<-nums[1:(length(nums)-1)] 
  return(paste("nums and nums_x have been reset"))
}

reset_numeric_vector(intrusion_df)

## checking summary of numerics
summary(intrusion_df[,nums_x])

# ## view histograms
# par(mfrow=c(3,3))
# for (n in nums){
#   hist(intrusion_df[,n]
#        ,main=paste("First Look Histogram of ",colnames(intrusion_df)[n],sep='')
#        ,xlab=''
#        ,col="light blue"
#        #,breaks=100
#   )
# }
# par(mfrow=c(1,1))
###########################################################

## Data Cleaning 
###########################################################

## this data set is already very clean.  Expect most of your time
## to be spent here.  expect most data to be in horrible shape
## when you first get it -- not in terms of the value it contains
##, but in how hard it is to get at it.
cleanIP<-function(ip){
  gsub("[^0-9.]", "", ip)
}

intrusion_df$srcip<-ip_to_numeric(cleanIP(intrusion_df$srcip))
intrusion_df$dstip<-ip_to_numeric(cleanIP(intrusion_df$dstip))

reset_numeric_vector(intrusion_df)
###########################################################

## Transformations on Native Numerics 
###########################################################

## Here I run histograms with 5 different transformations 
## to look for ones that approximate a normal distribution shape
#
## TIP: Consider researching more options to include
## and refactoring the code to cycle through them without the
## copy/paste -- come up with re-usable code to add to your
## toolbox.
par(mfrow=c(2,3)) # plots formatted to 2 rows, 3 columns
for (n in nums_x){
  hist(intrusion_df[,n]
       ,main=paste(" Untransformed Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=50
  )
  hist(log1p(intrusion_df[,n])
       ,main=paste("Log1p: Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=50
  )

  hist(sqrt(intrusion_df[,n])
       ,main=paste("Sqrt: Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=50
  )
  hist((intrusion_df[,n])^(1/3)
       ,main=paste("cubeRoot: Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=50
  )
  hist(intrusion_df[,n]^2
       ,main=paste("Square: Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=50
  )
  hist(1/(intrusion_df[,n])
       ,main=paste("recip: Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=50
  )
}
par(mfrow=c(1,1))

## correlation plots using 'nums' column index
M_tmp_orig<-cor(intrusion_df[,nums])
corrplot(M_tmp_orig,title="Correlation Matrix: Untransformed")

## here are the transformation choices (a bit subjective)
log1p_me<-c("Dintpkt","Sintpkt","Djit","Sjit","smeansz","Dpkts","Spkts","Dload","Sload"
            ,"dloss","sloss","dbytes","sbytes","dur","dsport","ct_dst_src_ltm")
sqrt_me<-c("ct_src_ ltm","ct_dst_ltm","dmeansz")
recip_me<-c("ackdat","synack","tcprtt")
scaled_collection<-c(log1p_me,sqrt_me,recip_me)

log1p_me_v<-which(colnames(intrusion_df) %in% log1p_me)
sqrt_me_v<-which(colnames(intrusion_df) %in% sqrt_me)
recip_me_v<-which(colnames(intrusion_df) %in% recip_me)

## we need to check to see if our desired transformations are possible
## eg, negative values cannot be log transformed.
summary(intrusion_df[,log1p_me_v])
summary(intrusion_df[,sqrt_me_v])
summary(intrusion_df[,recip_me_v])

for (c in log1p_me_v){
  intrusion_df[,c]<-log1p(intrusion_df[,c])
  intrusion_df[,c]<-scales::rescale(intrusion_df[,c],to=c(0,1))
}

for (c in sqrt_me_v){ # same idea as above but throwing functions into functions
  intrusion_df[,c]<-scales::rescale(sqrt(intrusion_df[,c]),to=c(0,1))
}

for (c in recip_me_v){
  intrusion_df[,c]<-scales::rescale((1/(ifelse(intrusion_df[,c]==0,0.00000001,intrusion_df[,c]))),to=c(0,1)) # dividing by 0 is not recommended
}

## check summaries as sanity check
summary(intrusion_df[,log1p_me_v])
summary(intrusion_df[,sqrt_me_v])
summary(intrusion_df[,recip_me_v]) 



## We now scale the numerics that were not transformed, and therefore not scaled yet
not_scaled<-vector()

## we only need to scale the numerics that go below 0 or above 1
for (ns in nums_x){
  if (summary(intrusion_df[,ns])[1]<0 | summary(intrusion_df[,ns])[6]>1){
    not_scaled<-c(not_scaled,ns)
  }
}

## we scale the remaining numerics (we had only treated the vars
## that had been transformed)
for (ns in not_scaled){
  intrusion_df[,ns]<-scales::rescale(intrusion_df[,ns],to=c(0,1)) 
}

# check to make sure ALL numeric features now have a range of 0 to 1
summary(intrusion_df[,nums_x]) ## looks good

## time to check to see if we improved our numeric features
M_tmp_scaled<-cor(intrusion_df[,nums])
corrplot(M_tmp_scaled,title="Correlation Matrix: Transformed Numerics") # conclusions?

## let's look at the difference in the histograms
par(mfrow=c(3,3))
for (n in nums_x){
  hist(intrusion_df[,n]
       ,main=paste("Transformed Histogram of Numerics:",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       ,breaks=100
  )
}
par(mfrow=c(1,1))
###########################################################

## Transform All Categoricals Into Numerics
###########################################################
## Research Topic: What is the difference between one-hot encoded
## and dummy variables?  What are the pros and cons?
#
## Note: You do not want to one-hot encode or 
## create dummy variables for too many levels.
## consider zipcodes.  Do you really want to create
## 42,000 features out of one?
#
## use your subject matter knowledge to group them
## into fewer levels; ideally 4-5 or less.
#
## (Related Research Topic: Curse of Dimensionality)
#
## here I naively reduce the options in the categoricals
## with no consideration for what the data represent
## whenever you can, leverage your knowledge in search of data
## that better explains the variance
#
cats<-which(unlist(lapply(intrusion_df, is.character))) 
for (cat in cats){ # cat = cats[1]
  tmp_df<-as.data.frame(table(intrusion_df[,cat]))
  tmp_df$Var1<-as.character(tmp_df$Var1)
  tmp_df<-tmp_df[order(tmp_df$Freq,decreasing=T),]
  nmbr_levels<-length(unique(intrusion_df[,cat]))
  total_hits<-sum(tmp_df$Freq,na.rm = T)
  tmp_df$pct_total<-tmp_df$Freq/rep(total_hits,nmbr_levels)
  levels_at_or_above_5_pct_total<-unique(tmp_df[which(tmp_df$pct_total>=0.05),1])
  
  tmp_column_v<-intrusion_df[,cat]
  for (rearrange in 1:length(levels_at_or_above_5_pct_total)){ # rearrange=1
    indx_tmp<-which(tmp_column_v==levels_at_or_above_5_pct_total[rearrange])
    tmp_column_v[indx_tmp]=rearrange
  }
  tmp_column_v<- as.integer(tmp_column_v) # this will force all the non_usables to null
  tmp_column_v[which(is.na(tmp_column_v))]=0
  tmp_column_v<- as.character(tmp_column_v) 
  intrusion_df[,cat]<-tmp_column_v
}



dummied<-dummies::dummy.data.frame(intrusion_df)
###########################################################

## Impute all Nulls With Mean
###########################################################
## Research Topic: what other options does one have for imputation?
## Tip #1: Research the MICE algorithm
## Tip #2: Research Bayesian PCA 
## Tip #3: Think through the logistics of implementing the various approaches 
## in a production system

for (c in 1:length(dummied)){ # c=1
  nulls_index<-which(is.na(dummied[,c]))
  
  if (length(nulls_index)>0){
    mean_val<-mean(dummied[-nulls_index,c])
    dummied[nulls_index,c]=mean_val
  }
}


###########################################################

## Determine Best balancing Approach Via Tournament 
###########################################################
## Research Topic: Learn each of these balancing algorithms
## Experiment Suggestion: shoot out the testing algorithms
## we used randomForests, but what about svm or KNN?

### was run at 1/4 data so it would finish running 
# ubConf <- list(type=c("ubOver", "ubUnder", "ubSMOTE", "ubOSS", "ubCNN", "ubENN", "ubNCL", "ubTomek"), percOver=200, percUnder=200, k=2, perc=50, method="percPos", w=NULL)
# balancing_approach<-unbalanced::ubRacing(formula=Label~.,data=dummied,algo="randomForest"
#                              , ubConf=ubConf,ntree=5,positive=1)
# 
# winning_rebalancer<-balancing_approach$best

###########################################################

## Feature Engineering 
###########################################################
## Research Topic #1: interaction variables
## Research Topic #2: clustering - kmeans and hierarchical
## Research Topic #'s 3 & 4: colinearity and PCA
## Tip #1: try putting the less correlative features
## with colinearity issues all into a 3 dimensional PCA
## and then remove them from the features and add in 
## the 3 dimensions of PCA as replacement
## Research Topic #4: other feature engineering techniques


###########################################################

## Feature Selection Part 1
###########################################################
## Research Topic: learn the standard feature selection techniques
## associated with traditional linear regression
## Tip #1: learn how Boruta works; how is it an improvement?
## Tip #2: explore Genetic Algorithms for feature selection
## in R, library(GA) and library(caret)


## look at the correlation matrix
for (i in seq(from=1,to=60, by=27)){
  M_tmp<-cor(dummied[,c(i:(ifelse((length(dummied)-i)<35,(length(dummied)-1),(i+35))),length(dummied))])
  corrplot(M_tmp)
}

## going off of correlation strength first (see above)
useful_looking<-c("ct_state_ttl","Sload","Dpkts","Ltime","ct_dst_src_ltm"
                  ,"sttl","dbytes","prototcp","protounas","srcip","dsport"
                  ,"proto0","proto1","proto2","state1","state2","state3"
                  ,"service1","service2","dmeansz","smeansz","Stime","ct_state_ttl")

colinears<-c("Dload","Spkts","tcprtt","synack","ackdat","Stime","servicedns"
             ,"ct_dst_sport_ltm","ct_src_dport_ ltm","ct_dst_ltm","ct_srv_dst"
             ,"ct_srv_src","sbytes","stateINT","protoudp","dloss","Ltime")

pca_me_index<-which(colnames(dummied) %in% colinears)
###########################################################

## Feature Engineering
###########################################################
## Research Topic #1: interaction variables
## Research Topic #2: clustering - kmeans and hierarchical
## Research Topic #'s 3 & 4: colinearity and PCA
## Tip #1: try putting the less correlative features
## with colinearity issues all into a 3 dimensional PCA
## and then remove them from the features and add in 
## the 3 dimensions of PCA as replacement
## Research Topic #4: other feature engineering techniques
par(mfrow=c(1,1))
pca_3d<-stats::prcomp(x=dummied[,pca_me_index],rank.=3)
screeplot(pca_3d)
# biplot(pca_3d)
summary(pca_3d)

dummied<-dummied[,-pca_me_index]
pca_scores<-as.data.frame(pca_3d$x)

## now we scale the PCA
for (c in 1:3){
  pca_scores[,c]<-scales::rescale(pca_scores[,c],to=c(0,1))   
}

summary(pca_scores)
dummied<-cbind(dummied,pca_scores)



# the Label var is no longer at the end.  We move it to the front now
label_loc<-which(colnames(dummied)=="Label")
new_order<-c(label_loc,1:(label_loc-1),((label_loc+1):length(dummied)))

dummied<-dummied[,new_order]
label_loc<-which(colnames(dummied)=="Label")



###########################################################

## Feature Selection Part 2
###########################################################

## the Boruta algorithm does an excellent job picking attributes, but
## still works best when colinear attributes are removed first.
#
## we could alternatively get to a small set of features simply through
## looking over the correlation matrices above. (useful_looking)
## e.g., ...
# keep_me_index<-which(colnames(dummied) %in% useful_looking)
# dummied<-dummied[,c(label_loc,(length(dummied)-2):length(dummied),keep_me_index)]
## 
## But we will use Boruta, as shown below, using a sample of the rows
## to speed up the processing time.  Beware, this can take a long time.

Boruta_select<-Boruta(x=dummied[sample(size=10000,x=nrow(dummied)),-label_loc]
                      ,y=dummied[sample(size=10000,x=nrow(dummied)),label_loc]
                      ,doTrace=2
                      ,maxRuns=200)
print(Boruta_select)
plot(Boruta_select,title="Feature Importance")

selected_features_index<-which(Boruta_select$finalDecision=="Confirmed")
selected_features_index<-c(selected_features_index,length(dummied))

## reduce the dataset to only the dimensions we plan to use (the winners)
dummied<-dummied[,selected_features_index]
summary(dummied)
## start presentation commentary here with prepped data
## strong correlation matrix
M_dummied<-cor(dummied)

## original ordering
corrplot(M_dummied)

## re-ordered by hierarchical clustering
corrplot(M_dummied, order = "hclust", hclust.method = "ward.D2", addrect = 3)
save.image("~/Projects/intrusionDetection/borutaComplete.RData")
###########################################################

## Model Building
###########################################################
## break out train,test,validation
validation_pct<-0.2
train_pct<-.67
seed<-42
nmbr_bootstraps<-15 # this should generally be higher to stabilize

## build the scoreboard to keep track of the various results
## we initialize it as an empty shell
## and rbind() to it as we go until it is full
master_scoreboard<-as.data.frame(matrix(NA,ncol=11,nrow=0))
colnames(master_scoreboard)=c("algorithm","run","precision","recall","F1","lift","TP","TN","FP","FN","training_time")


## bootstrapping for model validation
## Research Topic: Cross-fold validation versus bootstrapping -- Pros and Cons?
for (bs in 1:nmbr_bootstraps){ # bs=1
  seed=seed+1
  set.seed(seed)
  print(paste("starting bootstrap # ",bs, " out of ",nmbr_bootstraps, sep=''))

  ## we first identify a selection of the data to use for our final validation
  validationindx<-sample(size=nrow(dummied)*validation_pct,x=nrow(dummied))
  
  ## we then seperate out the validation set; the exploration set will be further split into train and test
  validation_df<-dummied[validationindx,]
  exploration_df<-dummied[-validationindx,]
  
  ## get the column index for the label var
  label_loc<-1 # in this case we know the last column is the label; we could find the index by name as well
  
  rebalanced<-unbalanced::ubUnder(X=exploration_df[,-label_loc], Y=exploration_df[,label_loc]) 

  ## make sure it worked as expected
  summary(dummied$Label); summary(rebalanced$Y)

  rebalanced_df<-cbind(rebalanced$Y,rebalanced$X)
  
  ## create train and test sets
  trainindx<-sample(size=nrow(rebalanced_df)*train_pct,x=nrow(rebalanced_df))
  train_df<-rebalanced_df[trainindx,]
  test_df<-rebalanced_df[-trainindx,]
  
  ## Define training and test sets
  x_train <- train_df[,-label_loc]
  y_train <- train_df[,label_loc]
  x_test <- test_df[,-label_loc]
  y_test <- test_df[,label_loc]
  
  ## Prepare the xgboost data sets using the DMatrix format
  dtrain<-xgboost::xgb.DMatrix(as.matrix(x_train),label=y_train)
  dtest<-xgboost::xgb.DMatrix(as.matrix(x_test),label=y_test)
  dVal<-xgboost::xgb.DMatrix(as.matrix(validation_df[-label_loc]),label=validation_df[,label_loc])
  
  ################ xgboost approach first
  ## Research Topic: Learn about Precision, Recall, and F1 scores
  ## Research Topic: Learn the other scores/terminology related to 
  ## confusion matrices
  ## Advanced Research Topic: Weighted F scores/Fbeta scores
  
  ## this sets the propensity score required to predict in the positive case
  ## the higher the threshold, the higher the Precision score
  ## but at the expense of Recall.

  thresh_val<-0.5 
  
  ## here we will set the number of boosted trees we want in XGBoost
  nrounds_val=15
  
  ## Research Topic: how would you optimize parameters for xgboost?
  param <- list(max_depth = 3, eta = 0.3, silent = 0, nthread = 2, 
                objective = "binary:logistic", eval_metric = "auc")
  watchlist <- list(train = dtrain, eval = dtest)
  
  start_timer<-as.integer(format(Sys.time(),c("%H","%M", "%S")))
  xgb_mdl<-xgboost::xgb.train(param,dtrain,nrounds=nrounds_val,watchlist)
  end_timer<-as.integer(format(Sys.time(),c("%H","%M", "%S")))
  
  train_time=(((end_timer[1]*60)+end_timer[2]+(end_timer[3]/60)) - ((start_timer[1]*60)+start_timer[2]+(start_timer[3]/60)))
  
  xgb_pred<-predict(xgb_mdl,dtest) 
  
  ## use these test results to optimize parameters, prevent over-fitting
  xgb_test_results<-caret::confusionMatrix(as.factor(ifelse(xgb_pred>thresh_val,1,0)),as.factor(y_test),positive='1')
  
  ## predict on the validation set which was not balanced
  xgb_pred_v<-predict(xgb_mdl,dVal)
  xgb_val_results<-caret::confusionMatrix(as.factor(ifelse(xgb_pred_v>thresh_val,1,0)),as.factor(validation_df[,label_loc]),positive='1')
  
  ## prepare to plot the effects of using different threshold values
  tmp_xgb_precision_v<-vector()
  tmp_xgb_recall_v<-vector()
  tmp_xgb_f1_v<-vector()
  for (tmp_thresh in seq(from=0.01, to=0.99, by=0.01)){
    tmp_xgb_val_results<-caret::confusionMatrix(as.factor(ifelse(xgb_pred_v>tmp_thresh,1,0)),as.factor(validation_df[,label_loc]),positive='1')
    tmp_xgb_precision_v<-c(tmp_xgb_precision_v,tmp_xgb_val_results$byClass[5])
    tmp_xgb_recall_v<-c(tmp_xgb_recall_v,tmp_xgb_val_results$byClass[6])
    tmp_xgb_f1_v<-c(tmp_xgb_f1_v,tmp_xgb_val_results$byClass[7])
  }

  plot_thresh_xgb<-as.data.frame(cbind(seq(from=0.01, to=0.99,by=0.01),tmp_xgb_precision_v,tmp_xgb_recall_v,tmp_xgb_f1_v))
  
  par(mfrow=c(1,1))
  ## visualizing the effect of threshold values, relationship of Precision,Recall, F1
  plot(y=plot_thresh_xgb$tmp_xgb_precision_v,x=plot_thresh_xgb$V1,col="dark blue",ylim=c(0.85,1)
       ,xlab="thresholds",ylab="Scores",type='l',main="Effects Of Threshold Settings: XGBoost")
  points(y=plot_thresh_xgb$tmp_xgb_precision_v,x=plot_thresh_xgb$V1,col="dark blue",pch=20)
  points(y=plot_thresh_xgb$tmp_xgb_recall_v,x=plot_thresh_xgb$V1,col="orange",type='l')
  points(y=plot_thresh_xgb$tmp_xgb_recall_v,x=plot_thresh_xgb$V1,col="orange",pch=20)
  points(y=plot_thresh_xgb$tmp_xgb_f1_v,x=plot_thresh_xgb$V1,col="purple",type='l')
  points(y=plot_thresh_xgb$tmp_xgb_f1_v,x=plot_thresh_xgb$V1,col="purple",pch=20)
  legend("bottomright",c("Precision","Recall","F1")
         ,col=c("dark blue", "orange", "purple"),pch=c(20,20,20)
         )
  
  ## save results to the scoreboard
  tmp_master_scoreboard<-as.data.frame(matrix(NA,ncol=11,nrow=1))
  colnames(tmp_master_scoreboard)=c("algorithm","run","precision","recall","F1","lift","TP","TN","FP","FN","training_time")
  
  tmp_master_scoreboard$algorithm="xgboost"
  tmp_master_scoreboard$run=bs
  tmp_master_scoreboard$precision=xgb_val_results$byClass[5]
  tmp_master_scoreboard$recall=xgb_val_results$byClass[6]
  tmp_master_scoreboard$F1=xgb_val_results$byClass[7]
  tmp_master_scoreboard$lift=unname(xgb_val_results$overall[1]-xgb_val_results$overall[5])
  tmp_master_scoreboard$TP=xgb_val_results$table[4]
  tmp_master_scoreboard$TN=xgb_val_results$table[1]
  tmp_master_scoreboard$FP=xgb_val_results$table[2]
  tmp_master_scoreboard$FN=xgb_val_results$table[3]
  tmp_master_scoreboard$training_time=train_time
  
  master_scoreboard<-rbind(master_scoreboard,tmp_master_scoreboard)
  rm(tmp_master_scoreboard) 
  
  ## Research Topic: How would you optimize parameters for DBN?
  ## ALSO: Where would you expect to get the most ROI for your time -
  ## going back to transformations and scaling, feature selection,
  ## feature engineering - OR - parameter tweaking here?

  ## about to have a param tuning example
  ## want these values outside the if() statement
  ## so later code works even if the whole  if-then section is commented out


  
  if (bs==1){ ## we only want to optimize once, then re-use for each run
    start_nodes<-1
    start_layers<-1
    hidden_layers_selected<-rep.int(x=start_nodes,times=start_layers)  
    learningrate_val<-0.8
    numepochs_val<-1
    batchsize_val<-250
    scores_v<-vector()
    #
    ## to speed things up we can use a subset of the data
    ## but it needs to have enough explanation of variance 
    ## for the probability distributions to emerge.
    optimize_me_index<-sample(size=nrow(x_train)*0.1, x=nrow(x_train))
    train_indx<-sample(size=length(optimize_me_index)*0.75, x=length(optimize_me_index))

    ## here is a simplistic way to choose some parameters to illustrate 
    ## how one might want to approach the problem via experimentation
    #
    ## this is intentionally good enough to spark interest in the
    ## process and bad enough to invite code changes to get you interacting.
    #
    ## first, write a function that only takes params that you will want to 
    ## tweak.  Then, output the error or success value that scores
    ## how well the settings did.  Here, we choose lift of Accuracy
    ## over the No Information Rate.  The higher the better,
    ## so when we call it we are looking to optimize to
    ## a maximum, not a minimum (which is appropriate when
    ## the function outputs error.)
    #
    ## this function will necessarily have params defined outside itself
    ## which is awkward.  But it is imperative that only params
    ## be defined that can be optimized.  We use the '...' 
    ## param to allow for undeclared pass-ins of additional optimization params
    dbn_tune<-function(nbr_nodes=5,...){ 
      nbr_nodes<-round(nbr_nodes)
      nbr_layers<-round(start_layers)
      print(paste("# nodes = ",nbr_nodes," | # layers =  ",nbr_layers,sep=''))
      hidden_layers<-rep.int(x=nbr_nodes,times=nbr_layers)
      
      counter_no_improvement_line=4 # this way you can pass in overrides into the function via ,...
      counter_no_change_line=5
      
      dbn_mdl_tmp<-deepnet::dbn.dnn.train(x=as.matrix(x_train[train_indx,]),y=y_train[train_indx]
                                          ,numepochs = numepochs_val
                                          ,batchsize = batchsize_val
                                          ,learningrate = learningrate_val
                                          ,hidden = hidden_layers
                                          ,cd = 10
                                          ,output = "sigm"
                                          #,hidden_dropout = 0.05
                                          
      )
      tmp_dbn_test_pred<-nn.predict(dbn_mdl_tmp,as.matrix(x_train[-train_indx,]))
      tmp_dbn_results<-caret::confusionMatrix(as.factor(ifelse(tmp_dbn_test_pred>.5,1,0)),as.factor(y_train[-train_indx]),positive='1')
      score<-unname(tmp_dbn_results$overall[1]-tmp_dbn_results$overall[5])
      scores_v<<-c(scores_v,score)
      print(paste("lift over NIR = ",score,sep=''))
      print(tmp_dbn_results)
      if (scores_v[length(scores_v)]==score){
        counter_no_change<<-counter_no_change+1
        if (counter_no_change>1){
          print(paste("No improvement: increasing the number of epochs from ",numepochs_val," to ",numepochs_val+5,sep=''))
          numepochs_val<<-numepochs_val+5
        }
        if (counter_no_change>2){
          print(paste("No improvement: decreasing batchsize from ",batchsize_val," to ",ifelse(batchsize_val>50,batchsize_val-25,50),sep=''))
          batchsize_val<<-ifelse(batchsize_val>50,batchsize_val-25,50)
        }
        if (counter_no_change>3){
          print(paste("No improvement: increasing number of layers from ",start_layers," to ",start_layers+1,sep=''))
          start_layers<<-start_layers+1
        }
      }
      
      if (score>champion_score){
        champion_score<<-score
        champion_nodes<<-nbr_nodes
        champion_layers<<-nbr_layers
        counter_no_change<<-0
        counter_no_improvement<<-0
      } 
      
      if(length(scores_v)>1){
        if (scores_v[length(scores_v)-1]>=scores_v[length(scores_v)]){
          counter_no_improvement<<-counter_no_improvement+1
          if (counter_no_improvement == counter_no_improvement_line | counter_no_change == counter_no_change_line){
            print("Optimization Exhausted")
            break
          }
        }
      }

      print(score)
      return(score)
    }
    
    counter_no_improvement<-0
    counter_no_change<-0
    champion_nodes<-start_nodes
    champion_layers<-start_layers
    champion_score<-0
    champion_score<-dbn_tune(nbr_nodes=start_nodes, nbr_layers=start_layers)
    optimize(f=dbn_tune,c(1,75),maximum=T) ## Genetic Algorithms (GA) allow for multiple params at once
    hidden_layers_selected<-rep.int(x=champion_nodes,times=champion_layers)
    
    plot(scores_v,main="Optimization Path",col="blue",pch=20,xlab="iteration")
    points(scores_v,col="blue",type='l',lty=1)
  }
  
  
  ## train the DBN model (Deep Belief Network)
  start_timer<-as.integer(format(Sys.time(),c("%H","%M", "%S")))
  dbn_mdl<-deepnet::dbn.dnn.train(x=as.matrix(x_train),y=y_train
                         ,numepochs = numepochs_val
                         ,batchsize = batchsize_val
                         ,learningrate = learningrate_val
                         ,hidden = hidden_layers_selected
                         ,cd = 10
                         ,output = "sigm"
                         #,hidden_dropout = 0.1
                         
  )
  end_timer<-as.integer(format(Sys.time(),c("%H","%M", "%S")))
  train_time=(((end_timer[1]*60)+end_timer[2]+(end_timer[3]/60)) - ((start_timer[1]*60)+start_timer[2]+(start_timer[3]/60)))
  
  
  dbn_test_pred<-nn.predict(dbn_mdl,as.matrix(x_test))
  dbn_results<-caret::confusionMatrix(as.factor(ifelse(dbn_test_pred>thresh_val,1,0)),as.factor(y_test),positive='1')
  
  dbn_val_pred<-nn.predict(dbn_mdl,as.matrix(validation_df[,-1], to=c(0,1)))
  dbn_val_results<-caret::confusionMatrix(as.factor(ifelse(dbn_val_pred>thresh_val,1,0)),as.factor(validation_df[,label_loc]),positive='1')
  
  
  ## prepare to plot the effects of using different threshold values for DBN 
  tmp_dbn_precision_v<-vector()
  tmp_dbn_recall_v<-vector()
  tmp_dbn_f1_v<-vector()
  for (tmp_thresh in seq(from=0.01, to=0.99, by=0.01)){
    tmp_dbn_val_results<-caret::confusionMatrix(as.factor(ifelse(dbn_val_pred>tmp_thresh,1,0)),as.factor(validation_df[,label_loc]),positive='1')
    tmp_dbn_precision_v<-c(tmp_dbn_precision_v,tmp_dbn_val_results$byClass[5])
    tmp_dbn_recall_v<-c(tmp_dbn_recall_v,tmp_dbn_val_results$byClass[6])
    tmp_dbn_f1_v<-c(tmp_dbn_f1_v,tmp_dbn_val_results$byClass[7])
  }
  
  ## visualizing the effect of threshold values, relationship of Precision,Recall, F1 on DBN 
  plot_thresh_dbn<-as.data.frame(cbind(seq(from=0.01, to=0.99,by=0.01),tmp_dbn_precision_v,tmp_dbn_recall_v,tmp_dbn_f1_v))
  ## visualizing the effect of threshold values, relationship of Precision,Recall, F1
  plot(y=plot_thresh_dbn$tmp_dbn_precision_v,x=plot_thresh_dbn$V1,col="dark blue",ylim=c(0.85,1)
       ,xlab="thresholds",ylab="Scores",type='l',main="Effects Of Threshold Settings: Deep Belief Network")
  points(y=plot_thresh_dbn$tmp_dbn_precision_v,x=plot_thresh_dbn$V1,col="dark blue",pch=20)
  points(y=plot_thresh_dbn$tmp_dbn_recall_v,x=plot_thresh_dbn$V1,col="orange",type='l')
  points(y=plot_thresh_dbn$tmp_dbn_recall_v,x=plot_thresh_dbn$V1,col="orange",pch=20)
  points(y=plot_thresh_dbn$tmp_dbn_f1_v,x=plot_thresh_dbn$V1,col="purple",type='l')
  points(y=plot_thresh_dbn$tmp_dbn_f1_v,x=plot_thresh_dbn$V1,col="purple",pch=20)
  legend("bottomright",c("Precision","Recall","F1")
         ,col=c("dark blue", "orange", "purple"),pch=c(20,20,20)
  )
  tmp_master_scoreboard<-as.data.frame(matrix(NA,ncol=11,nrow=1))
  colnames(tmp_master_scoreboard)=c("algorithm","run","precision","recall","F1","lift","TP","TN","FP","FN","training_time")
  
  tmp_master_scoreboard$algorithm="dbn NN"
  tmp_master_scoreboard$run=bs
  tmp_master_scoreboard$precision=dbn_val_results$byClass[5]
  tmp_master_scoreboard$recall=dbn_val_results$byClass[6]
  tmp_master_scoreboard$F1=dbn_val_results$byClass[7]
  tmp_master_scoreboard$lift=unname(dbn_val_results$overall[1]-dbn_val_results$overall[5])
  tmp_master_scoreboard$TP=dbn_val_results$table[4]
  tmp_master_scoreboard$TN=dbn_val_results$table[1]
  tmp_master_scoreboard$FP=dbn_val_results$table[2]
  tmp_master_scoreboard$FN=dbn_val_results$table[3]
  tmp_master_scoreboard$training_time=train_time
  
  master_scoreboard<-rbind(master_scoreboard,tmp_master_scoreboard); rm(tmp_master_scoreboard)
  
  ## ensemble scores
  ## Research Topic: How else might you ensemble predictions?
  
  ens_pred<-(xgb_pred_v + dbn_val_pred)/2 # a simplistic form of ensembling
  
  ens_results<-caret::confusionMatrix(as.factor(ifelse(ens_pred>thresh_val,1,0)),as.factor(validation_df[,label_loc]),positive='1')
  
  tmp_master_scoreboard<-as.data.frame(matrix(NA,ncol=10,nrow=1))
  colnames(tmp_master_scoreboard)=c("algorithm","run","precision","recall","F1","lift","TP","TN","FP","FN")
  
  tmp_master_scoreboard$algorithm="Ensembled xgb-dbn"
  tmp_master_scoreboard$run=bs
  tmp_master_scoreboard$precision=ens_results$byClass[5]
  tmp_master_scoreboard$recall=ens_results$byClass[6]
  tmp_master_scoreboard$F1=ens_results$byClass[7]
  tmp_master_scoreboard$lift=unname(ens_results$overall[1]-ens_results$overall[5])
  tmp_master_scoreboard$TP=ens_results$table[4]
  tmp_master_scoreboard$TN=ens_results$table[1]
  tmp_master_scoreboard$FP=ens_results$table[2]
  tmp_master_scoreboard$FN=ens_results$table[3]
  tmp_master_scoreboard$training_time=NA
  
  master_scoreboard<-rbind(master_scoreboard,tmp_master_scoreboard); rm(tmp_master_scoreboard) 
  
  ## prepare to plot the effects of using different threshold values for DBN 
  tmp_ens_precision_v<-vector()
  tmp_ens_recall_v<-vector()
  tmp_ens_f1_v<-vector()
  for (tmp_thresh in seq(from=0.01, to=0.99, by=0.01)){
    tmp_ens_val_results<-caret::confusionMatrix(as.factor(ifelse(ens_pred>tmp_thresh,1,0)),as.factor(validation_df[,label_loc]),positive='1')
    tmp_ens_precision_v<-c(tmp_ens_precision_v,tmp_ens_val_results$byClass[5])
    tmp_ens_recall_v<-c(tmp_ens_recall_v,tmp_ens_val_results$byClass[6])
    tmp_ens_f1_v<-c(tmp_ens_f1_v,tmp_ens_val_results$byClass[7])
  }
  
  ## visualizing the effect of threshold values, relationship of Precision,Recall, F1 on ens 
  plot_thresh_ens<-as.data.frame(cbind(seq(from=0.01, to=0.99,by=0.01),tmp_ens_precision_v,tmp_ens_recall_v,tmp_ens_f1_v))
  ## visualizing the effect of threshold values, relationship of Precision,Recall, F1
  plot(y=plot_thresh_ens$tmp_ens_precision_v,x=plot_thresh_ens$V1,col="dark blue",ylim=c(0.85,1)
       ,xlab="thresholds",ylab="Scores",type='l',main="Effects Of Threshold Settings: Ensembled XGB and DBN")
  points(y=plot_thresh_ens$tmp_ens_precision_v,x=plot_thresh_ens$V1,col="dark blue",pch=20)
  points(y=plot_thresh_ens$tmp_ens_recall_v,x=plot_thresh_ens$V1,col="orange",type='l')
  points(y=plot_thresh_ens$tmp_ens_recall_v,x=plot_thresh_ens$V1,col="orange",pch=20)
  points(y=plot_thresh_ens$tmp_ens_f1_v,x=plot_thresh_ens$V1,col="purple",type='l')
  points(y=plot_thresh_ens$tmp_ens_f1_v,x=plot_thresh_ens$V1,col="purple",pch=20)
  legend("bottomright",c("Precision","Recall","F1")
         ,col=c("dark blue", "orange", "purple"),pch=c(20,20,20)
  )
}

par(mfrow=c(1,3))
boxplot(master_scoreboard$precision[which(master_scoreboard$algorithm=="xgboost")],col="purple",ylab="Precision",xlab="xgboost",ylim=c(0.9,0.92))
boxplot(master_scoreboard$precision[which(master_scoreboard$algorithm=="dbn NN")],col="orange",ylab="Precision",xlab="DBN",ylim=c(0.9,0.92))
boxplot(master_scoreboard$precision[which(master_scoreboard$algorithm=="Ensembled xgb-dbn")],col="dark green",ylab="Precision",xlab="Ensemble xgb + DBN",ylim=c(0.9,.92))

boxplot(master_scoreboard$recall[which(master_scoreboard$algorithm=="xgboost")],col="purple",ylab="Recall",xlab="xgboost",ylim=c(0.995,1))
boxplot(master_scoreboard$recall[which(master_scoreboard$algorithm=="dbn NN")],col="orange",ylab="Recall",xlab="DBN",ylim=c(0.995,1))
boxplot(master_scoreboard$recall[which(master_scoreboard$algorithm=="Ensembled xgb-dbn")],col="dark green",ylab="Recall",xlab="Ensemble xgb + DBN",ylim=c(0.995,1))

boxplot(master_scoreboard$F1[which(master_scoreboard$algorithm=="xgboost")],col="purple",ylab="F1",xlab="xgboost",ylim=c(0.95,0.96))
boxplot(master_scoreboard$F1[which(master_scoreboard$algorithm=="dbn NN")],col="orange",ylab="F1",xlab="DBN",ylim=c(0.95,0.96))
boxplot(master_scoreboard$F1[which(master_scoreboard$algorithm=="Ensembled xgb-dbn")],col="dark green",ylab="F1",xlab="Ensemble xgb + DBN",ylim=c(0.95,0.96))

par(mfrow=c(1,2))
boxplot(master_scoreboard$training_time[which(master_scoreboard$algorithm=="xgboost")],col="purple",ylab="Model Training Time (minutes)",xlab="xgboost",ylim=c(0,60))
boxplot(master_scoreboard$training_time[which(master_scoreboard$algorithm=="dbn NN")],col="orange",ylab="Model Training Time (minutes)",xlab="DBN",ylim=c(0,60))
par(mfrow=c(1,1))



###########################################################

## Final Steps
###########################################################
#
## Key Question #1: How does the model perform?
#
## Recommended Answer: communicate with the business
## in terms of both the average performance (the expected value)
## after enough bootstrapping runs for it to be dependable
## AND variability, which is shown in the boxplots.
#
## A common mistake is to only communicate the average
## scores without controlling executive expectations
## re: reasonable variability.  Give a 95% confidence interval
## or some kind of upper and lower bound to prevent
## snatching defeat from the jaws of victory

## Model Performance in terms of F1 score @ the threshold setting used
F1_performance<-mean(master_scoreboard$F1)

se <- function(x) sqrt(var(x)/length(x))
F1_se<-se(master_scoreboard$F1)
F1_ci<-c(F1_performance-F1_se, F1_performance+F1_se) 
names(F1_ci)=c("lower bound","upper bound")
print(paste("This model has an F1 score of ",round(F1_performance,3), " +- ",round(F1_se,4)," @ 95% confidence",sep=''))

####### Advanced Business Application
## assume a FP costs the company $200
## assume a FN costs the company $2000
## assume a TN gains the company $0
## asume a TP gains the company $2000

## Research Topic: How would you determine these
## values in your company
TP_bounty  <- 200
TN_bounty  <- 0
FP_penalty <- 0 - 20
FN_penalty <- 0 - 200

results_v<-vector()
champion_result<-0
champion_thresh<-0

counter=0
for (tmp_thresh in seq(from=0.00, to=1, by=0.01)){
  counter=counter+1
  tmp_ens_val_results<-confusionMatrix(as.factor(ifelse(ens_pred>tmp_thresh,1,0)),as.factor(validation_df[,label_loc]),positive='1')
  TP_result<-tmp_ens_val_results$table[4]*TP_bounty
  TN_result<-tmp_ens_val_results$table[1]*TN_bounty
  FP_result<-tmp_ens_val_results$table[2]*FP_penalty
  FN_result<-tmp_ens_val_results$table[3]*FN_penalty
  final_result<-TP_result+TN_result+FP_result+FN_result
  results_v<-c(results_v,final_result)
  if (final_result>champion_result){
    champion_result<<-final_result
    champion_thresh<<-counter/100
  }
}

par(mfrow=c(1,1))
plot(results_v,col="dark blue",xlab="thresholds",ylab="Business Results",type='l',main="Effects Of Threshold Settings on Business Value")
abline(v=champion_thresh*100,col="dark gray",lty=3)
text(x=(champion_thresh*100)+3,y=0,champion_thresh,col="dark gray")

dev.off()
save.image("~/Projects/intrusionDetection/CompleteBig.RData")
