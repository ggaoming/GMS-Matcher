# -×- coding:utf-8 -*-
import cv2
import os
from gms_matcher import gms_matcher
import time

NUM_ORB_FEATURE = 10000  # number of feature in ORB

class GMS():

    def __init__(self):
        self.orb = cv2.ORB(NUM_ORB_FEATURE)
        pass

    def orb_detect(self, img):
        keypoints = self.orb.detect(img, None)
        keypoints, descriptions = self.orb.compute(img, keypoints)
        return keypoints, descriptions
        pass

    def draw_result(self, img1, img2, kp1, kp2, matches):
        height1, width1, _ = img1.shape
        height2, width2, _ = img2.shape
        output_img = np.concatenate((img1, img2), axis=1)
        for p in kp1:
            p = p.pt
            cv2.circle(output_img, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)
        for p in kp2:
            p = p.pt
            cv2.circle(output_img, (int(p[0] + width1), int(p[1])), 2, (0, 255, 0), 2)
        for i in range(min(len(matches), 100)):
            pt1 = kp1[matches[i].queryIdx].pt
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = kp2[matches[i].trainIdx].pt
            pt2 = (int(pt2[0] + width1), int(pt2[1]))
            cv2.line(output_img, pt1, pt2, (255, 0, 0), 1)
        cv2.imshow('src', output_img)
        cv2.waitKey()
        pass

    def process(self, img_path1, img_path2):
        """
        :param img_path1: path for image 
        :param img_path2: path for image
        :return: 
        """
        assert os.path.exists(img_path1) and os.path.exists(img_path2)

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        img1 = cv2.resize(img1, (500, 500))
        img2 = cv2.resize(img2, (500, 500))

        kp1, des1 = self.orb_detect(img1)
        kp2, des2 = self.orb_detect(img2)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches_all = matcher.match(des1, des2)
        gms = gms_matcher(kp1, img1.shape, kp2, img2.shape, matches_all)
        num_inliers, inlier_mask = gms.GetInlierMask(False, False)
        # self.draw_result(img1, img2, kp1, kp2, matches_all)
        matches_gms = []

        for i in range(len(inlier_mask)):
            if inlier_mask[i]:
                matches_gms.append(matches_all[i])
        self.draw_result(img1, img2, kp1, kp2, matches_gms)

    def camera_test(self):
        img1 = None
        img2 = None
        cap = cv2.VideoCapture(0)
        while True:
            state, frame = cap.read()

            if img1 is not None:
                start_time = time.time()
                img2 = frame.copy()
                img1 = cv2.resize(img1, (480, 480))
                img2 = cv2.resize(img2, (480, 480))

                kp1, des1 = self.orb_detect(img1)
                kp2, des2 = self.orb_detect(img2)

                matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches_all = matcher.match(des1, des2)
                time1 = time.time()
                gms = gms_matcher(kp1, img1.shape, kp2, img2.shape, matches_all)
                num_inliers, inlier_mask = gms.GetInlierMask(True, True)
                time2 = time.time()
                print time1 - start_time, time2 - time1
                # self.draw_result(img1, img2, kp1, kp2, matches_all)
                matches_gms = []

                for i in range(len(inlier_mask)):
                    if inlier_mask[i]:
                        matches_gms.append(matches_all[i])
                self.draw_result(img1, img2, kp1, kp2, matches_gms)
            else:
                cv2.imshow('src', cv2.resize(frame, (500, 500)))
            key = 0xFF & cv2.waitKey(10)
            if key == ord('a'):
                img1 = frame.copy()
                pass
            elif key == 27:
                break
                pass
            elif key == 255:
                continue
            else:
                print key






if __name__ == '__main__':
    app = GMS()
    app.process('data/nn_left.jpg', 'data/nn_right.jpg')
    #app.camera_test()
    pass