###########################################################
##  @author: James Guymon 
##  @date: Sep 23 2019
##  @email: james@jamesguymon.com
##  @Twitter: @JamesGuymon
#
## Forward
###########################################################
#
## This document is first and foremost pedagogical,
## and therefore verbose in places so the processes are clear
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
###########################################################

## Getting Started
###########################################################
#
## creation of the directories and the downloading the files 
## should work automatically.  The following instructions explain
## how to set up the project manually
#
## Step 1: create a new directory for this project and save this file into it
#
## Step 2: create 'data_in' and 'viz' folders inside that directory
#
## Step 3: save the .csv files from 
## https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/ 
## into data_in
#
## Step 4: adjust line 56 (below) to point to your directory.
## an easy way to do it in RStudio via the headers nav 
## is: Session -> Set Working Directory -> To Source File Location

## this option prevents scientific notation in plots
options(scipen=999)

## this sets some graphics params at the onset
par(mfrow=c(1,1))

######## directory/file management

## define paths
#### the user should set this
projectRootDir<-"~/Projects/intrusionDetection" 
inputFiles<-"/data_in/"

## create folders we will use shortly, but only if they do not exist
ifelse(!dir.exists(file.path(projectRootDir,inputFiles)), dir.create(file.path(projectRootDir, inputFiles)), FALSE)

## set our work directory so we have relative paths in the code
setwd(projectRootDir)

## list of the files we will use
nb_15_filenames<-c("UNSW-NB15_1.csv"
                   ,"UNSW-NB15_2.csv"
                   ,"UNSW-NB15_3.csv"
                   ,"UNSW-NB15_4.csv"
                   ,"NUSW-NB15_features.csv"
                   ,"NUSW-NB15_GT.csv"
                   ,"UNSW-NB15_LIST_EVENTS.csv")

## the download url structure we will modify with filenames
nb15_urls_template<-"https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20CSV%20Files&files="

## download each file and save to our /data_in folder, but only if we haven't grabbed them already

downloaded_already<-list.files(paste(".",inputFiles,sep=''))
download_me<-which(!(nb_15_filenames %in% downloaded_already))

if (length(download_me)>0){
  for (file in 1:length(nb_15_filenames[download_me])){ 
    print(paste("downloading ",nb_15_filenames[download_me][file],sep=''))
    nb15_url<-paste(nb15_urls_template,nb_15_filenames[download_me][file],sep='')
    destination_file<-paste(".",inputFiles,nb_15_filenames[download_me][file],sep='')
    download.file(url = nb15_url,destfile = destination_file)
  }  
} else {
  print("All NB15 files already downloaded.")
}

## NOTE: This project brings the data to us - a common
## paradigm, however an increasingly archaic one.  As
## we move to bigger data sets the approach is to leave
## the data where it is and bring the processing to it.
#
## RESEARCH TOPIC: Hadoop

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
              , "iptools", "bit64","dplyr","caret","scatterplot3d")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}

# ## commented out because they are not used here, but recommended highly
# ## Check them out!
# extended_packages <- c("mice", "bpca", "GA", "e1071","tidyverse")
# if (length(setdiff(extended_packages, rownames(installed.packages()))) > 0) {
#   install.packages(setdiff(extended_packages, rownames(installed.packages())))  
# }

library(data.table) ## a useful library for loading in data
library(dplyr) # a library that makes data manipulation easier

## Data Visualiation Aids
library(Amelia) # good viz package for missing data
library(network) # other use as well, but useful in visualizing networks
library(corrplot) # beautiful correlation matrix plots
library(igraph) # network plots -- also library(network) is useful
library(scatterplot3d) # visualize 3D scatterplots

# ## imputation tools you should know
# library(mice) # an historically important package for null imputation; not computationally efficient
# library(bpca) # an approach worth exploring, generally preferred by the author

# Data Transformation
library(unbalanced) # a package of all required tools to rebalance data sets
library(dummies) # a package to transform categorical variables into numerics
library(scales) # helpful to recale data into format that can be transformed

# Feature Selection
library(Boruta) # highly recommended gem
# library(GA) # revisit when ready for an advanced technique with expansive uses

# # ML algorithms
# library(keras)
# library(tensorflow); install_tensorflow()
library(deepnet) # a low-learning-curve deep learning package
library(xgboost) # a great swiss-army knife algorithm that can do both regression and classification
library(randomForest) # another staple in the ML toolkit; we load it to use in the re-balancing tournament
# library(e1071) # a library with many ML algorithms to try

# Model Measurement
library(caret) ## lots of uses here

## of particular use in this intrusion dataset
library(iptools) # transforms ip addresses into a numeric form
require(bit64) # allows 64 bit integers at import

# # helper functions section

## taken from my personal toolkit
makeDensityPlots<-function(num,title="Untitled Density Plots",plots.per.line=4,nmbr.pages=1){
  old.par<-par(no.readonly=TRUE)
  nmbr.attributes<-dim(num)[2]
  nmbr.required.rows<-ceiling(nmbr.attributes/plots.per.line) 
  par(mar=c(1,1,1,1),mfrow=c((nmbr.required.rows/nmbr.pages),plots.per.line))
  for (i in 1:dim(num)[2]){
    hist(num[,i],freq=FALSE,main=names(num)[i],xlab=names(num)[i],col="#99ccff",border="#669999")
    lines(density(num[,i], na.rm = T,adjust=1.5),col="#00802d")
    lines(density(num[,i], na.rm = T,adjust=2.5),lty="dotted",col="#334d00")
    rug(quantile(num[,i],probs=seq(from=0.05,to=0.95,by=0.01),na.rm=TRUE),col="#9999ff",lwd=0.75,side=3)
    rug(quantile(num[,i],probs=seq(from=0.25,to=0.75,by=0.01),na.rm=TRUE),col="#0000b3",lwd=1,side=3)
    rug(median(num[,i],na.rm=TRUE),col="#ff751a",lwd=3.85,side=3)
    rug(num[,i],col="#ffcccc",lwd=.45,side=1)
    rug(quantile(num[,i],probs=seq(from=0.0015,to=0.9985,by=0.01),na.rm=TRUE),col="#ff6666",lwd=0.75,side=1)
    rug(quantile(num[,i],probs=seq(from=0.16,to=0.84,by=0.01),na.rm=TRUE),col="#ff0000",lwd=1,side=1)
    rug(mean(num[,i],na.rm=TRUE),col="#ffff4d",lwd=2.85,side=1)
  }
  title(title,outer=TRUE)
  on.exit(par(old.par))
}
###########################################################

## Data Load
###########################################################
## Tip: Practice reading in messy data
## such as crawled html.  Challenge yourself until you 
## can read in anything.
#
## Research Topic: Streaming data within big data.

column_names<-data.table::fread("./data_in/NUSW-NB15_features.csv",stringsAsFactors=F,data.table=F)
ground_truth<-data.table::fread("./data_in/NUSW-NB15_GT.csv",stringsAsFactors=F,data.table=F)
list_events<-data.table::fread("./data_in/UNSW-NB15_LIST_EVENTS.csv",stringsAsFactors=F,data.table=F)

## Often we need to construct large data sets from smaller files
## it is best to do this programmatically rather than one by one
#
## here is an elegant way to do it, followed by alternative code that is easier to track

dta_filenames<-paste("./data_in/UNSW-NB15_",1:4,".csv",sep='')
intrusion_df<-do.call(rbind,lapply(dta_filenames,fread,stringsAsFactors=F,header=F,data.table=F,col.names = column_names$Name))


# # ## The same process as above, less efficient, but easier to follow
# for (chunk in 1:4){
#   tmp_df<-fread(paste("./data_in/UNSW-NB15_",chunk,".csv",sep=''),stringsAsFactors=F,header=F,data.table=F)
# 
#   if (chunk==1){
#     intrusion_df<-as.data.frame(matrix(NA,ncol=length(tmp_df),nrow=0))
#   }
#   intrusion_df<-rbind(intrusion_df,tmp_df); rm(tmp_df)
# }
# 
# colnames(intrusion_df)=column_names$Name

## both 'sport' and 'dsport' shoud be integers
## but came in as characters due to '-' chars
## after reviewing the problematic rows
## we can coerce to integer, which will generate nulls
## but the '-' char represented NULL anyway.

intrusion_df$sport<-as.integer(intrusion_df$sport)
intrusion_df$dsport<-as.integer(intrusion_df$dsport)

# ## if one wanted to use the 64bit integers this would be helpful
# intrusion_df$stcpb<-bit64::as.integer64(intrusion_df$stcpb)
# intrusion_df$dtcpb<-bit64::as.integer64(intrusion_df$dtcpb)

## we remove columns we will not use
## readers may choose to keep and use them
remove_me<-c("attack_cat","stcpb","dtcpb")

intrusion_df<-intrusion_df[,-which(colnames(intrusion_df) %in% remove_me)]

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

## simple helper function for inside dplyr aggregation
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
## initialize empty vectors first, then fill
nums<-vector(); nums_x<-vector()

## we will need to refresh the vars from time to time
## so we make it a function to avoid copy/paste
reset_numeric_vector<-function(df){
  nums<<-which(unlist(lapply(df, is.numeric))) 
  nums_x<<-nums[1:(length(nums)-1)] 
  return(paste("nums and nums_x have been reset"))
}

reset_numeric_vector(intrusion_df)
# print(nums) ## prints a list of the numerics

## checking summary of numerics
summary(intrusion_df[,nums_x])

## view histograms for all numerics
par(mfrow=c(3,3))
for (n in nums){
  hist(intrusion_df[,n]
       ,main=paste("First Look Histogram of ",colnames(intrusion_df)[n],sep='')
       ,xlab=''
       ,col="light blue"
       #,breaks=100
  )
}
par(mfrow=c(1,1))

## the following is an example of how one's toolkit might evolve over time
makeDensityPlots(intrusion_df[visualize_me,nums],title="First Look: Numerics",plots.per.line=2,nmbr.pages=3)

###########################################################

## Data Cleaning 
###########################################################

## this data set is already very clean.  Expect most of your time
## to be spent here.  expect most data to be in horrible shape
## when you first get it -- not in terms of the value it contains
##, but in how hard it is to get at it.

## you'll notice that srcip and dstip have more characters
## in them apart from the '.' which will prevent
## the ip_to_numeric() function to work
## we have to clean the data first

cleanIP<-function(ip){
  gsub("[^0-9.]", "", ip)
}

intrusion_df$srcip<-ip_to_numeric(cleanIP(intrusion_df$srcip))
intrusion_df$dstip<-ip_to_numeric(cleanIP(intrusion_df$dstip))

reset_numeric_vector(intrusion_df) ## there are 2 new numerics, so an update is in order
###########################################################

## Transformations 
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
## the ?'s occur where correlation cannot be calculated -- they are NULLs
corrplot(M_tmp_orig,title="Correlation Matrix: Untransformed")

## here are the transformation choices (a bit subjective)
## to recreate this, review the correlation plots
## and place the column names of the attributes in the respective
## transformation type you want to commit to
## add more transformation types as well!  just modify
## the code above to include more transformations to test
## then add a bucket for them below, eg, new_trans<-c(x1,x2,xn)
log1p_me<-c("Dintpkt","Sintpkt","Djit","Sjit","smeansz","Dpkts","Spkts","Dload","Sload"
            ,"dloss","sloss","dbytes","sbytes","dur","dsport","ct_dst_src_ltm")
sqrt_me<-c("ct_src_ ltm","ct_dst_ltm","dmeansz")
recip_me<-c("ackdat","synack","tcprtt")
scaled_collection<-c(log1p_me,sqrt_me,recip_me)

## a bit inefficent to have both the column names above
## and the vector of indices here, but the workflow demands it
## first you make your choices (above) then prepare the code
## to be cleaner (below)

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
# we initialize and empty vector then fill it shortly
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

## One-Hot Encoding
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

ohed<-dummies::dummy.data.frame(intrusion_df)

## Q: Why not transform AFTER converting categoricals to numerics?
## Aren't we missing an opportunity to transform the one-hot
## encoded variables?
#
## A: Although the one-hot encoded variables are numerics/integers
## they are functioning as Booleans.  Transforming them would be 
## non-sensical.
###########################################################

## Null Treatment
###########################################################
## Research Topic: what other options does one have for imputation?
## Tip #1: Research the MICE algorithm
## Tip #2: Research Bayesian PCA 
## Tip #3: Think through the logistics of implementing the various approaches 
## in a production system
## Tip #4: Consider throwing away the problematic rows

for (c in 1:length(ohed)){ # c=1
  nulls_index<-which(is.na(ohed[,c]))
  
  if (length(nulls_index)>0){
    mean_val<-mean(ohed[-nulls_index,c])
    ohed[nulls_index,c]=mean_val
  }
}

###########################################################

## Feature Engineering 
###########################################################
#
## Research Topic #1: interaction variables
## Research Topic #2: clustering - kmeans and hierarchical
## Research Topic #'s 3 & 4: colinearity and PCA
## Tip #1: try putting the less correlative features
## with colinearity issues all into a 3 dimensional PCA
## and then remove them from the features and add in 
## the 3 dimensions of PCA as replacement
## Research Topic #4: other feature engineering techniques
#

## look at the correlation matrices
for (i in seq(from=1,to=60, by=27)){
  M_tmp<-cor(ohed[,c(i:(ifelse((length(ohed)-i)<35,(length(ohed)-1),(i+35))),length(ohed))])
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

pca_me_index<-which(colnames(ohed) %in% colinears)

## Research Topic #1: interaction variables
## Research Topic #2: clustering - kmeans and hierarchical
## Research Topic #'s 3 & 4: colinearity and PCA
## Tip #1: try putting the less correlative features
## with colinearity issues all into a 3 dimensional PCA
## and then remove them from the features and add in 
## the 3 dimensions of PCA as replacement
## Research Topic #4: other feature engineering techniques
par(mfrow=c(1,1))
pca_3d<-stats::prcomp(x=ohed[,pca_me_index],rank.=3)
screeplot(pca_3d, col="light blue",type='lines',main="Screeplot on Collinear Features in PCA")

## curiosity check-in of the pca
scatterplot3d(x=pca_3d$x[visualize_me,1]
              ,y=pca_3d$x[visualize_me,2]
              ,z=pca_3d$x[visualize_me,3]
              ,angle=120
              ,type='h'
              ,pch=20
              ,color=ifelse(ohed$Label[visualize_me]==0,"dark green","red")
              ,xlab = "PCA Dimension 1"
              ,ylab = "PCA Dimension 2"
              ,zlab = "PCA Dimension 3"
              ,main="Scatterplot of PCA in 3 Dimensions")
legend("bottomleft",legend=c("Intrusion","Non-intrusion")
       ,col=c("red","dark green")
       ,pch=c(20,20))


cor(pca_3d$x[,1],ohed$Label)
cor(pca_3d$x[,2],ohed$Label)
cor(pca_3d$x[,3],ohed$Label)
summary(pca_3d)

## inspired investigation based upon the PCA scattplot
## we load the initial data set again and get the intrusions
## as disctinct types again to plot with the scatterplot3d

intrusion_df2<-do.call(rbind,lapply(dta_filenames,fread,stringsAsFactors=F,header=F,data.table=F,col.names = column_names$Name))


attacks<-intrusion_df2$attack_cat
attacks[which(attacks=='')]="Normal" ## we name normal traffic rather than leave it NULL
attacks[which(attacks=='Backdoors')]="Backdoor" ## the data set have ambiguous labeling -- we fix it
summary(attacks)

attacks<-as.factor(attacks)

intrusion.only<-which(ohed$Label==1)
summary(attacks[intrusion.only])
proportions.attacks<-summary(attacks)/nrow(ohed)
# write.csv(proportions.attacks,"attack_proportions.csv")

## test out the color pallete
palette<-rainbow(length(levels(attacks[intrusion.only])))
colors<-palette[factor(attacks[intrusion.only])]
plot(seq.int(1:length(levels(attacks[intrusion.only]))),col=palette
     ,xlab="Intrusion Types",ylab='',main="Color Pallete", pch=20,cex=4)
legend("bottomright",legend=levels(attacks[intrusion.only])
       ,col=palette
       ,pch=rep(20,length(levels(attacks[intrusion.only])))
)
 
## draw a scatterplot3d highighting intrusion
scatterplot3d(x=pca_3d$x[intrusion.only,1]
              ,y=pca_3d$x[intrusion.only,2]
              ,z=pca_3d$x[intrusion.only,3]
              ,angle=120
              ,type='h'
              ,pch=20
              ,color=colors
              ,xlab = "PCA Dimension 1"
              ,ylab = "PCA Dimension 2"
              ,zlab = "PCA Dimension 3"
              ,main="Scatterplot of PCA in 3 Dimensions: Colored by Intrusion Type")
legend("bottomleft",legend=levels(attacks[intrusion.only])
       ,col=palette
       ,pch=rep(20,length(levels(attacks[intrusion.only])))
)

## draw another scatterplot3d highighting intrusion, rotated
scatterplot3d(x=pca_3d$x[intrusion.only,1]
              ,y=pca_3d$x[intrusion.only,2]
              ,z=pca_3d$x[intrusion.only,3]
              ,angle=358
              ,type='h'
              ,pch=20
              ,color=colors
              ,xlab = "PCA Dimension 1"
              ,ylab = "PCA Dimension 2"
              ,zlab = "PCA Dimension 3"
              ,main="Scatterplot of PCA in 3 Dimensions: Colored by Intrusion Type")
legend("bottomleft",legend=levels(attacks[intrusion.only])
       ,col=palette
       ,pch=rep(20,length(levels(attacks[intrusion.only])))
)

# ### feature creation experiment for 'generic'
## Research Topic: How might you create a feature to represent
## what you see in the plots for generic intrusions?
#
## Other intrusion types?

## we look at IPs to see if there is a pattern there
ips<-intrusion_df2$srcip
summary(ips)

simplifyIP<-function(char){
  char<-sub(pattern="^(.*)[.].*","\\1",x=char[1])  
  return(char)
}

ips<-sapply(ips,FUN=simplifyIP)

ips<-as.factor(ips)
levels(ips)

ips.palette<-rainbow(length(levels(ips[intrusion.only])))
ips.colors<-ips.palette[factor(ips[intrusion.only])]
scatterplot3d(x=pca_3d$x[intrusion.only,1]
              ,y=pca_3d$x[intrusion.only,2]
              ,z=pca_3d$x[intrusion.only,3]
              ,angle=140+90
              ,type='h'
              ,pch=20
              ,color=ips.colors
              ,xlab = "PCA Dimension 1"
              ,ylab = "PCA Dimension 2"
              ,zlab = "PCA Dimension 3"
              ,main="Scatterplot of PCA in 3 Dimensions: Colored by Source IP")
legend("bottomleft",legend=levels(ips[intrusion.only])
       ,col=ips.palette
       ,pch=rep(20,length(levels(ips[intrusion.only])))
       )


################################
ohed<-ohed[,-pca_me_index]
pca_scores<-as.data.frame(pca_3d$x)

## now we scale the PCA
for (c in 1:3){
  pca_scores[,c]<-scales::rescale(pca_scores[,c],to=c(0,1))   
}

summary(pca_scores)
ohed<-cbind(ohed,pca_scores)

# the Label var is no longer at the end.  We move it to the front now
label_loc<-which(colnames(ohed)=="Label")
new_order<-c(label_loc,1:(label_loc-1),((label_loc+1):length(ohed)))

ohed<-ohed[,new_order]
label_loc<-which(colnames(ohed)=="Label")

###########################################################

## Feature Selection
###########################################################
#
## Research Topic: learn the standard feature selection techniques
## associated with traditional linear regression
## Tip #1: learn how Boruta works; how is it an improvement?
## Tip #2: explore Genetic Algorithms for feature selection
## in R, library(GA) and library(caret)
#
## the Boruta algorithm does an excellent job picking attributes, but
## still works best when colinear attributes are removed first.
#
## we could alternatively get to a small set of features simply through
## looking over the correlation matrices above. (useful_looking)
## e.g., ...
# keep_me_index<-which(colnames(ohed) %in% useful_looking)
# ohed<-ohed[,c(label_loc,(length(ohed)-2):length(ohed),keep_me_index)]
## 
## But we will use Boruta, as shown below, using a sample of the rows
## to speed up the processing time.  Beware, this can take a long time.

Boruta_select<-Boruta(x=ohed[sample(size=10000,x=nrow(ohed)),-label_loc]
                      ,y=ohed[sample(size=10000,x=nrow(ohed)),label_loc]
                      ,doTrace=2
                      ,maxRuns=200)
print(Boruta_select)
plot(Boruta_select,title="Feature Importance")

selected_features_index<-which(Boruta_select$finalDecision=="Confirmed")
selected_features_index<-c(selected_features_index,length(ohed))

## reduce the dataset to only the dimensions we plan to use (the winners)
ohed<-ohed[,selected_features_index]
summary(ohed)
## start presentation commentary here with prepped data
## strong correlation matrix
M_ohed<-cor(ohed)

## original ordering
corrplot(M_ohed)

## re-ordered by hierarchical clustering
corrplot(M_ohed, order = "hclust", hclust.method = "ward.D2", addrect = 5)
save.image("~/Projects/intrusionDetection/borutaComplete.RData")
###########################################################

## Balancing
###########################################################
## Research Topic: Learn each of these balancing algorithms
## Experiment Suggestion: shoot out the testing algorithms
## we used randomForests, but what about svm or KNN?
#
## We only run the tournament now to determine the best approach
## The actual balancing is applied inside the modeling section
## because we need to first break out the dataset into
## a non-affected validation set, and then balanced train/test
## sets for training and optimizing the models.
ubConf <- list(type=c("ubOver", "ubUnder", "ubSMOTE", "ubOSS", "ubCNN", "ubENN", "ubNCL", "ubTomek"), percOver=200, percUnder=200, k=2, perc=50, method="percPos", w=NULL)
balancing_approach<-unbalanced::ubRacing(formula=Label~.,data=ohed,algo="randomForest"
                                         , ubConf=ubConf,ntree=5,positive=1)
## we save the winning algorithm to a variable
winning_rebalancer<-balancing_approach$best

## Important note:  This dataset doesn't appear to need balancing
## as we look at the results.  Often the tournament will result
## in the unbalanced approach winning, and even when CNN wins
## it is by a slim margin that doesn't justify the modification
## which comes with its own risk (it alters the data set)
#
## For this reason we choose under-sampling despite it not
## being the winner.  It has the benefit of resulting in 
## a 50/50 split of positive and negative cases in the target variable
## where CNN only increases the proportion by ~0.02.
#
###########################################################

## Model Building
###########################################################
#
## Suggested Tool: MLFlow, created by databricks
## https://mlflow.org/
#
## it offers solutions for collaborating and tracks parameters
## and outcomes.
#
# ### quick check-in with full categorization model
# ## 10 fold cross-validation, no balancing, go too far
xgb.explore<-xgb.cv(data=as.matrix(ohed[,-1]),label=attacks,nrounds=25,max_depth=8,nfold=10,eta=1,nthread=4,prediction=T)
print(xgb.explore, verbose=T)
# 
## plot train vs test to show overfitting
plot(y=xgb.explore$evaluation_log$train_rmse_mean, x=xgb.explore$evaluation_log$iter
     ,main="Overfitting: Divergence of Train and Test Results",ylab="RMSE",xlab="# of Trees: XGBoost"
     ,pch=20,col="dark orange")
points(y=xgb.explore$evaluation_log$train_rmse_mean, x=xgb.explore$evaluation_log$iter,type='l',col="dark orange")
points(y=(xgb.explore$evaluation_log$train_rmse_std*2)+xgb.explore$evaluation_log$train_rmse_mean, x=xgb.explore$evaluation_log$iter,type='l',col="orange",lty=2)
points(y=xgb.explore$evaluation_log$train_rmse_mean-(xgb.explore$evaluation_log$train_rmse_std*2), x=xgb.explore$evaluation_log$iter,type='l',col="orange",lty=2)
points(y=xgb.explore$evaluation_log$test_rmse_mean, x=xgb.explore$evaluation_log$iter,pch=20,col="dark blue")
points(y=xgb.explore$evaluation_log$test_rmse_mean, x=xgb.explore$evaluation_log$iter,type='l',col="dark blue")
points(y=(xgb.explore$evaluation_log$test_rmse_std*2)+xgb.explore$evaluation_log$test_rmse_mean, x=xgb.explore$evaluation_log$iter,type='l',col="blue",lty=2)
points(y=xgb.explore$evaluation_log$test_rmse_mean-(xgb.explore$evaluation_log$test_rmse_std*2), x=xgb.explore$evaluation_log$iter,type='l',col="blue",lty=2)
legend("bottomleft",legend=c("train","2 sd","test","2 sd"),col=c("dark orange","orange","dark blue","blue"),pch=c(20,NA,20,NA),lty=c(1,2,1,2))

## quick 75/25 split of train and test
experiment.train<-sample(size=nrow(ohed)*.75,x=nrow(ohed))

## xgboost model
xgb.full.pred<-xgboost(data=as.matrix(ohed[experiment.train,-1]),label=attacks[experiment.train],nrounds=15,max_depth=8,eta=1,nthread=4)
xgb.save(xgb.full.pred,"./all_intrusion_types_xgb.model")

pred.xgb<-predict(xgb.full.pred,as.matrix(ohed[-experiment.train,-1]))

pred.level<-round(pred.xgb); pred.level[which(pred.level<1)]=1

full.intrusion.explore<-confusionMatrix(as.factor(levels(attacks)[pred.level]),attacks[-experiment.train],mode="everything")
full.intrusion.explore

# write.csv(full.intrusion.explore$overall, "fullIntrusion_results_overall.csv")
# write.csv(full.intrusion.explore$table, "fullIntrusion_results_table.csv")
# write.csv(full.intrusion.explore$byClass, "fullIntrusion_results_class.csv")

## Research question: How would one balance when predicting multiple factor levels?  Try it.

## Now back to predicting the 2-level factor: intrusion vs non-intrusion
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
  validationindx<-sample(size=nrow(ohed)*validation_pct,x=nrow(ohed))
  
  ## we then seperate out the validation set; the exploration set will be further split into train and test
  validation_df<-ohed[validationindx,]
  exploration_df<-ohed[-validationindx,]
  
  ## get the column index for the label var
  label_loc<-1 # in this case we know the last column is the label; we could find the index by name as well
  
  rebalanced<-unbalanced::ubUnder(X=exploration_df[,-label_loc], Y=exploration_df[,label_loc]) 
  
  ## make sure it worked as expected
  summary(ohed$Label); summary(rebalanced$Y)
  
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
  
  if (bs==1){ ## we only want to optimize once, then re-use for each run
    start_nodes<-1
    start_layers<-1
    hidden_layers_selected<-rep.int(x=start_nodes,times=start_layers)  
    learningrate_val<-0.8
    numepochs_val<-1
    batchsize_val<-250
    scores_v<-vector()
    
    counter_no_improvement<-0
    counter_no_change<-0
    champion_nodes<-start_nodes
    champion_layers<-start_layers
    champion_score<-0
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
      improvement_exhausted = FALSE
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
      
      if (length(scores_v)>1){
        if (scores_v[length(scores_v)-1]>=scores_v[length(scores_v)]){
          counter_no_improvement<<-counter_no_improvement+1
          if (counter_no_improvement == counter_no_improvement_line | counter_no_change == counter_no_change_line){
            print("Optimization Exhausted")
            improvement_exhausted = TRUE
            break
          }
        if (improvement_exhausted){
          break
          }
        }
      if (improvement_exhausted){
        break
        }
      }
      
      print(score)
      return(score)
    }
    

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
  
  par(mfrow=c(1,1))
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


xgb.save(xgb_mdl,"./binary_intrusion_types_xgb.model")



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

## Reinforcement Learning
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
## assume a FP costs the company $20
## assume a FN costs the company $200
## assume a TN gains the company $0
## asume a TP gains the company $200

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

# results_v[55] - results_v[50]

par(mfrow=c(1,1))
plot(results_v,col="dark blue",xlab="thresholds",ylab="Business Results",type='l',main="Effects Of Threshold Settings on Business Value")
abline(v=champion_thresh*100,col="dark gray",lty=3)
text(x=(champion_thresh*100)+3,y=0,champion_thresh,col="dark gray")

# dev.off()
save.image("~/Projects/intrusionDetection/CompleteBig.RData")
# write.csv(master_scoreboard,"~/Projects/intrusionDetection/scoreboard.csv")
# write.csv(as.data.frame(Boruta_select$ImpHistory),"~/Projects/intrusionDetection/Boruta_select.csv")
