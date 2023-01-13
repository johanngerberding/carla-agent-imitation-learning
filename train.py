import h5py
import glob 
import cv2 
import matplotlib.pyplot as plt 


def main(): 
    path = "/data/AgentHuman/SeqTrain"
    train_files = glob.glob(path + "/*.h5")    
    print(f"Number of training files: {len(train_files)}")
    # per h5 file we have 200 images   
    test = h5py.File(train_files[0], 'r')
    print(type(test))
    print(test.keys())
    print(type(test['rgb']))
    print(type(test['targets']))
    print(test["rgb"].shape)
    print(test["targets"].shape)
    rgb = test["rgb"][0]
    print(rgb.shape)
    print(type(rgb)) 
    cv2.imshow('image', rgb)
    target = test["targets"][0]
    print(target)


if __name__ == "__main__":
    main()