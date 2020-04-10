# Soccer Robot Perception

Packages Required:
* python3 == 3.6.9
* torch == 1.40
* pillow == 7.0.0
* numpy == 1.18.2
* opencv-python == 4.2.0.34
* torchvision == 0.5.0


1. Run main.py which provided with following arguments=

    --help,--learning_rate=,--batchSize=,-epochs=,--TVLossWeightDetect,--TVLossWeightSegment,--saveGeneratedImages
*     --help  for displaying help
*     --learning_rate for setting desired learning rate
*     --batchSize for setting batch size
*     --epochs for setting epochs
*     --TVLossWeightDetect for setting the weight for total variation loss for detection task
*     --TVLossWeightsegment for setting the weight for total variation loss for segmnetation task
*     --saveGeneratedImages to save the inputs,ground truths as well as ouputs from training on test dataset
*     Example Use: python main.py --learning_rate=0.01 --saveGeneratedImages; this command will set learning rate as sepcified and save images generated from model
     



2. Run getResults.py to get the results on test data

3 .Run instanceInference.py to get instance results .Specify the folder name which contains images to be tested

*     --folderName for specifying the folder which contains the files
*     Example Use: python instanceInference.py --folderName="./sample_images"





We have used dataprepare.py to remove the missing files from dataset as well as create a dataset with representation
of variety of inputs in train,test and validate dataset.