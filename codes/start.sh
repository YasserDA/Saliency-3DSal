# cloning Project From  Git

# echo 'cloning Project From  Git ...'

# git clone <URL>

##################################################################################


# Creating The Folder architecture

echo 'Creating The Folder architecture ...'

mkdir ./WORKSPACE
mkdir ./WORKSPACE/DATASET
mkdir ./WORKSPACE/DATASET/annotation
mkdir ./WORKSPACE/DATASET/annotation/images
mkdir ./WORKSPACE/DATASET/annotation/maps
mkdir ./WORKSPACE/BATCH
mkdir ./WORKSPACE/BATCH/TR_BATCH
mkdir ./WORKSPACE/BATCH/VGG-16
mkdir ./WORKSPACE/BATCH/GT_BATCH
mkdir ./WORKSPACE/DATA/GT_DATA
mkdir ./WORKSPACE/DATA/TR_DATA
mkdir ./WORKSPACE/TEST
mkdir ./WORKSPACE/TEST/annotation
mkdir ./WORKSPACE/TEST/result
mkdir ./WORKSPACE/TRAINED_MODEL

##################################################################################

#Downloading the Dataset and unziping it to location

echo 'Downloading the Dataset and unziping it to location...'

#Download the dataset you want: DHF1K, UFC-SPORT, DAVIS...etc
#prepare it and make the video frames in the folder '' ./WORKSPACE/DATASET/annotation/images ' and the Ground truth maps in the folder: '' ./WORKSPACE/DATASET/annotation/images ''
#################################################################################


echo 'Downloading the 3DSal models...'

#Download the models from the Github links and make them in the folder: '' ./WORKSPACE/TRAINED_MODEL''
#################################################################################


