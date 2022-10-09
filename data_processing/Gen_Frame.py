#adopted from https://github.com/saakur/EventSegmentation/blob/master/preprocessVideo.py

import cv2, os, numpy


def Gen_Breakfast_Frame(video_path,frameOutPath):
    if not os.path.exists(frameOutPath):
        os.mkdir(frameOutPath)
    vidPaths_Subject = [str(os.path.join(video_path, f) + '/') for f in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, f))]

    vidPaths = [str(os.path.join(vidPath, f) + '/') for vidPath in vidPaths_Subject for f in os.listdir(vidPath) if os.path.isdir(os.path.join(vidPath, f))]

    for camPath in vidPaths:
        # print camPath
        vidFilePaths = [os.path.join(camPath, f) for f in os.listdir(camPath) if
                        os.path.isfile(os.path.join(camPath, f)) and f.endswith('.avi')]
        if not vidFilePaths:
            continue
        for vidFile in vidFilePaths:
            cap = cv2.VideoCapture(vidFile)
            totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print vidFile, totalFrame
            subID = vidFile.split('/')[-2]
            vidName = vidFile.split('/')[-1].split('.')[0].split('_')
            vidName.insert(1, subID)
            vidName = '_'.join(vidName)
            outFilePrefix = os.path.join(frameOutPath, vidName)
            print(vidName, outFilePrefix, totalFrame)
            if not os.path.exists(outFilePrefix):
                print("Frame out path %s does not exist... Creating..." % outFilePrefix)
                os.makedirs(outFilePrefix)
            currFrame = 0
            while (True):
                try:
                    ret, frame = cap.read()
                except:
                    continue
                if not ret:
                    break

                currFrame += 1

                outFileName = os.path.join(outFilePrefix, "Frame_%06d.jpg" % currFrame)
                # print outFileName
                cv2.imwrite(outFileName, frame)
            # break
            cap.release()
        # break

    # break
    print("Finished processing frames!!")

from ops.os_operation import mkdir
def Gen_Salad_Frame(video_path,frameOutPath):
    if not os.path.exists(frameOutPath):
        os.mkdir(frameOutPath)

    listfiles=[os.path.join(video_path,x) for x in os.listdir(video_path) if ".avi" in x]

    for vidFile in listfiles:
        # print camPath
        cap = cv2.VideoCapture(vidFile)
        totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print vidFile, totalFrame
        vidName = os.path.split(vidFile)[1][:-4]

        outFilePrefix = os.path.join(frameOutPath, vidName)
        mkdir(outFilePrefix)
        print("video name %s, output path %s"%(vidName,outFilePrefix))

        currFrame = 0
        while (True):
            try:
                ret, frame = cap.read()
            except:
                continue
            if not ret:
                break

            currFrame += 1

            outFileName = os.path.join(outFilePrefix, "Frame_%06d.jpg" % currFrame)
                # print outFileName
            cv2.imwrite(outFileName, frame)
            # break
        cap.release()
        # break

    # break
    print("Finished processing frames!!")

def Gen_INRIA_Frame(video_path,frameOutPath):
    if not os.path.exists(frameOutPath):
        os.mkdir(frameOutPath)

    listfiles=[x for x in os.listdir(video_path)]

    for item in listfiles:
        tmp_dir=os.path.join(video_path,item)
        if not os.path.isdir(tmp_dir):
            continue
        tmp_dir=os.path.join(tmp_dir,"videos")
        listfile1=[x for x in os.listdir(tmp_dir) if ".mpg" in x]
        for item1 in listfile1:
            vidFile=os.path.join(tmp_dir,item1)
            # print camPath
            cap = cv2.VideoCapture(vidFile)
            totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print vidFile, totalFrame
            vidName = os.path.split(vidFile)[1][:-4]

            outFilePrefix = os.path.join(frameOutPath, vidName)
            mkdir(outFilePrefix)
            print("video name %s, output path %s"%(vidName,outFilePrefix))

            currFrame = 0
            while (True):
                try:
                    ret, frame = cap.read()
                except:
                    continue
                if not ret:
                    break

                currFrame += 1

                outFileName = os.path.join(outFilePrefix, "Frame_%06d.jpg" % currFrame)
                # print outFileName
                cv2.imwrite(outFileName, frame)
                # break
            cap.release()
        # break

    # break
    print("Finished processing frames!!")
