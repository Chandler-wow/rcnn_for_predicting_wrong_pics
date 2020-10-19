import sys
import cv2
def selective_search(img):
    # speed-up using multithreads
    cv2.setUseOptimized(True)  # 使用优化
    cv2.setNumThreads(4)  # 开启多线程计算

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rects[:,2] += rects[:,0]
    rects[:,3] += rects[:,1]
    return rects[:,[1,0,3,2]]