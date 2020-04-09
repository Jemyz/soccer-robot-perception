import os,fnmatch,shutil
directory = "C:\\Users\Sarah Khan\Downloads\small_data\detection"
directory_source = "C:\\Users\Sarah Khan\Downloads\\bigcuda5.informatik.uni-bonn.de+8686\\blob"
folder = "train"
count =0
flag=0
num=1
chance =1
for file in os.listdir(os.path.join(directory_source,"dataset")):
    if fnmatch.fnmatch(file, "*.jpg") or fnmatch.fnmatch(file,"*.png"):
        flag=0
        for file1 in os.listdir(os.path.join(directory_source,"dataset")):
            base = os.path.splitext(file)[0]
            if(file1 == base+".xml" ):
                flag=1
                try:
                    shutil.copy(os.path.join(directory_source,"dataset", file),
                            os.path.join(directory,folder,"input", file))
                except EnvironmentError:
                    print("Error happened")
                    print(file)
                    exit()
                else:
                    print("OK")
                try:
                    shutil.copy(os.path.join(directory_source,"dataset", file1),
                            os.path.join(directory, folder, "output", file1))
                except EnvironmentError:
                    print("Error happened")
                    print(file1)
                    exit()
                else:
                    print("OK")

                print(file)
                print(file1)
                count = count+1
                if(count%7==0 and folder=="train"):
                    count =0
                    folder = "validate"
                    if(num == 1):
                        num =2
                    else:
                        num = 1
                    break
                if(count%num==0 and folder=="validate"):
                    count = 0
                    folder = "test"
                    break
                if (count % num == 0 and folder == "test"):
                    count = 0
                    folder = "train"
                    break
                break




