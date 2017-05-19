# -*- coding:utf-8 -*-
import cv2
import math
from collections import defaultdict
THRESH_FACTOR = 6


class gms_matcher(object):

    def __init__(self, point1, size1, point2, size2, matches):
        """
        :param point1: img1's keypoints
        :param size1: img1's size 
        :param point2: img2's keypoints
        :param size2: img2's size
        :param matches: matches index
        """
        # normalize points to 0.0~1.0 [x/width, y/height]
        self.mvP1 = self.normalize_points(point1, size1)
        self.mvP2 = self.normalize_points(point2, size2)
        self.mNumberMatches = len(matches)  # pair match result length
        self.mvMatches = self.convert_matches(matches)  # match result
        self.mGridSizeLeft = [20, 20]  # left(original) image grid size initialize
        self.mGridSizeRight = [0, 0]  # right(new) image gird size
        self.mGridNumberLeft = self.mGridSizeLeft[0] * self.mGridSizeLeft[1]
        self.mRotationPatterns = [
            [1, 2, 3,
             4, 5, 6,
             7, 8, 9],
            [4, 1, 2,
             7, 5, 3,
             8, 9, 6],
            [7, 4, 1,
             8, 5, 2,
             9, 6, 3],
            [8, 7, 4,
             9, 5, 1,
             6, 3, 2],
            [9, 8, 7,
             6, 5, 4,
             3, 2, 1],
            [6, 9, 8,
             3, 5, 7,
             2, 1, 4],
            [3, 6, 9,
             2, 5, 8,
             1, 4, 7],
            [2, 3, 6,
             1, 5, 9,
             4, 7, 8]
            ]
        self.mScaleRatios = [1.0, 0.5, 1.0/math.sqrt(2.0), math.sqrt(2.0), 2.0]  # scales changes
        # [map<int, int>] index: grid_index_left  dict: key:grid_index_right value: number points matched idx_l to idx_r
        self.mMotionStatistics = [defaultdict(int)]
        self.mCellPairs = []  # [int]  index: grid_index_left, value: grid_index_right
        self.mNumberPointsInPerCellLeft = []  # [int, ]
        self.mvbInlierMask = []  # bool
        # [(left_idx, right_idx)]
        self.mvMatchPairs = []  # [(int, int)]
        pass

    def convert_matches(self, matches):
        """
        convert match result queryid and trainid to pairs format (int, int)
        :param matches: 
        :return: [[q_id0, t_id0], [q_id1, t_id1], [q_id2, t_id2]]
        """
        new_matches = [[i.queryIdx, i.trainIdx] for i in matches]
        return new_matches

    def normalize_points(self, points, img_size):
        """
        conver point x,y to x/width, y/height
        :param points:  [p1, p2, p3]
        :param img_size: height, width, channel
        :return: 
        """
        height, width, _ = img_size
        normalize_points = [[p.pt[0]*1.0 / width, p.pt[1]*1.0 / height] for p in points]
        return normalize_points

    def GetInlierMask(self, with_scale, with_rotation):
        """
        match result mask
        :param with_scale: width scale changes 
        :param with_rotation: width rotation changes
        :return: return the result with most number of match points
        """
        max_inlier = 0  # number of inlier
        vbInliers = []
        if (not with_scale) and (not with_rotation):  # without scale and rotation changes
            max_inlier = self.run(1, 0)
            return max_inlier, self.mvbInlierMask[:]

        if with_rotation and with_scale:  # scale and rotation
            for scale in range(5):  # value 0-4
                for rotation in range(1, 9):  # value 1-8
                    num_inlier = self.run(rotation, scale)
                    if num_inlier > max_inlier:
                        max_inlier = num_inlier
                        vbInliers = self.mvbInlierMask[:]
            return max_inlier, vbInliers

        if with_rotation and (not with_scale):  # only rotation
            for rotation in range(1, 9):
                num_inlier = self.run(rotation, 0)
                if num_inlier > max_inlier:
                    max_inlier = num_inlier
                    vbInliers = self.mvbInlierMask[:]
            return max_inlier, vbInliers

        if (not with_rotation) and with_scale:  # only scale
            for scale in range(5):
                num_inlier = self.run(1, scale)
                if num_inlier > max_inlier:
                    max_inlier = num_inlier
                    vbInliers = self.mvbInlierMask[:]
            return max_inlier, vbInliers

        return max_inlier, vbInliers
        pass

    def AssignMatchPairs(self, GridType):
        """
        using match points x,y find match pair's id in grid box
        :param GridType: 
        :return: 
        """
        for i in range(self.mNumberMatches):
            lp = self.mvP1[self.mvMatches[i][0]]  # x,y  match pairs
            rp = self.mvP2[self.mvMatches[i][1]]
            lgidx = int(self.GetGridIndexLeft(lp, GridType))  # index in grid
            rgidx = int(self.GetGridIndexRight(rp))
            self.mvMatchPairs[i][0] = lgidx
            self.mvMatchPairs[i][1] = rgidx
            self.mMotionStatistics[lgidx][rgidx] += 1
            self.mNumberPointsInPerCellLeft[lgidx] += 1

    def GetGridIndexLeft(self, pt, type_):
        """
        use x,y find the grid index for pt
        :param pt: (0.0-1.0, 0.0-1.0)
        :param type_:  1: move left_top, 2: move right_top, 3: move left_bottom, 4: move right_bottom
        :return: 
        """
        x = 0
        y = 0
        if type_ == 1:  # pt ( x/width, y/height) grid (20, 200)
            x = math.floor(pt[0] * self.mGridSizeLeft[0])
            y = math.floor(pt[1] * self.mGridSizeLeft[1])
        elif type_ == 2:
            x = math.floor(pt[0] * self.mGridSizeLeft[0] + 0.5)
            y = math.floor(pt[1] * self.mGridSizeLeft[1])
        elif type_ == 3:
            x = math.floor(pt[0] * self.mGridSizeLeft[0])
            y = math.floor(pt[1] * self.mGridSizeLeft[1] + 0.5)
        elif type_ == 4:
            x = math.floor(pt[0] * self.mGridSizeLeft[0] + 0.5)
            y = math.floor(pt[1] * self.mGridSizeLeft[1] + 0.5)
        if x >= self.mGridSizeLeft[0] or y >= self.mGridSizeLeft[1]:
            return -1
        else:
            return x + y * self.mGridSizeLeft[0]
        pass

    def GetGridIndexRight(self, pt):
        """
        use x,y find grid index for pt
        :param pt: 
        :return: 
        """
        x = math.floor(pt[0] * self.mGridSizeRight[0])
        y = math.floor(pt[1] * self.mGridSizeRight[1])
        return x + y * self.mGridSizeRight[0]
        pass

    def VerifyCellPairs(self, RotationType):
        CurrentRP = self.mRotationPatterns[RotationType - 1]
        for i in range(self.mGridNumberLeft):
            if len(self.mMotionStatistics[i]) == 0:
                self.mCellPairs[i] = -1
                continue
            max_num = 0
            for pf, ps in self.mMotionStatistics[i].items():
                if ps > max_num:
                    self.mCellPairs[i] = pf
                    max_num = ps
            if max_num <= 1:
                self.mCellPairs[i] = -1
                continue

            idx_grid_rt = self.mCellPairs[i]

            NB9_lt = self.GetNB9(i, self.mGridSizeLeft)
            NB9_rt = self.GetNB9(idx_grid_rt, self.mGridSizeRight)

            score = 0
            thresh = 0
            numpair = 0

            for j in range(9):
                ll = NB9_lt[j]
                rr = NB9_rt[CurrentRP[j] - 1];
                if ll == -1 or rr == -1:
                    continue
                score += self.mMotionStatistics[ll][rr]
                thresh += self.mNumberPointsInPerCellLeft[ll]
                numpair += 1
            if numpair != 0:
                thresh = THRESH_FACTOR * 1.0 * math.sqrt(thresh / numpair)
            else:
                thresh = 0

            if score < thresh:
                self.mCellPairs[i] = -2


        pass

    def GetNB9(self, idx, GridSize):
        NB9 = [-1 for _ in range(9)]
        idx_x = int(idx) % int(GridSize[0])
        idx_y = int(idx) / int(GridSize[0])
        for yi in [-1, 0, 1]:
            for xi in [-1, 0, 1]:
                idx_xx = idx_x + xi
                idx_yy = idx_y + yi
                if idx_xx < 0 or idx_xx >= GridSize[0] or idx_yy < 0 or idx_yy >= GridSize[1]:
                    continue
                NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize[0]
        return NB9

    def run(self, RotationType, Scale):

        self.mvbInlierMask = [False for _ in range(self.mNumberMatches)]  # * self.mNumberMatches
        for GridType in range(1, 5):
            self.mGridSizeRight[0] = int(self.mGridSizeLeft[0] * self.mScaleRatios[Scale])
            self.mGridSizeRight[1] = int(self.mGridSizeLeft[1] * self.mScaleRatios[Scale])
            self.mMotionStatistics = [defaultdict(int) for _ in range(self.mGridNumberLeft)]
            self.mCellPairs = [-1 for _ in range(self.mGridNumberLeft)]
            self.mNumberPointsInPerCellLeft = [0 for _ in range(self.mGridNumberLeft)]
            self.mvMatchPairs = [[0, 0] for _ in range(self.mNumberMatches)]
            self.AssignMatchPairs(GridType)
            self.VerifyCellPairs(RotationType)
            for i in range(self.mNumberMatches):
                if self.mCellPairs[self.mvMatchPairs[i][0]] == self.mvMatchPairs[i][1]:
                    self.mvbInlierMask[i] = True
        num_inlier = sum(self.mvbInlierMask)

        return num_inlier
        pass



if __name__ == '__main__':
    pass
