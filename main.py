# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 21:41:33 2020

@author: Sarah Khan
"""
import getopt
import sys
from train import train
def help():
    print("Options available")
    print("--help or -h: for help")
    print("--learning_rate for setting desired learning rate")
    print("--batchSize for setting batch size")
    print("--epochs for setting epochs")
    print("--TVLossWeightDetect for setting the weight for total variation loss for detection task")
    print("--TVLossWeightsegment for setting the weight for total variation loss for segmnetation task")
    print("--saveGeneratedImages to save the inputs,ground truths as well as ouputs from training on test dataset")
    print("Example Use: python main.py --learning_rate=0.01 --saveGeneratedImages; this command will set learning rate as sepcified and save images generated from model")
    sys.exit(0)
    
def main():
    #Assign some defaults
    learning_rate = 0.01
    batchSize = 20
    epochs = 150
    tvlossweightdetect = 0.000001
    tvlossweightsegment = 0.00001
    saveImages = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h",["help","learning_rate=","batchSize=","epochs=","TVLossWeightDetect=","TVLossWeightSegment=","saveGeneratedImages"])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
               help()
            elif opt in ("--learning_rate"):
               learning_rate = float(arg)
            elif opt in ("--batchSize"):
                batchSize = int(arg)
            elif opt in ("--epochs"):
                epochs = int(arg)
            elif opt in ("--TVLossWeightDetect"):
                tvlossweightdetect = float(arg)
            elif opt in ("--TVLossWeightSegment"):
                tvlossweightsegment = float(arg)
            elif opt in ("--saveGeneratedImages"):
                saveImages = True
            else:
                help()
        train(learning_rate,batchSize,epochs,tvlossweightdetect,tvlossweightsegment,saveImages)
    except:
        print("Check help")
    
    
    
if __name__ == "__main__":
    main()
