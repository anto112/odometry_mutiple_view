import numpy as np
import cv2
from config import CameraConfig


class MonoVisualOdometry(object):
    def __init__(self, img_file_path, pose_file_path, focal_length, camera_parameter, threshold=int(19.5)):
        """
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
            pose_file_path {str} -- File path that leads to true poses from image sequences

        Raises:
            ValueError --
        """
        self.threshold = threshold
        self.file_path = img_file_path
        self.cam = CameraConfig(camera_parameter)
        self.lk_param = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 30, 0.001))
        self.focal_length = focal_length
        self.pp = self.cam.pp
        self.point_reference = None
        self.point_now = None
        self.draw_x, self.draw_y = 0, 0
        self.R = np.zeros(shape=(3, 3))
        self.curr_r = None
        self.t = np.zeros(shape=(3, 3))
        self.curr_t = None
        self.id = 0
        self.previmage = None
        self.kMinNumFeature = 1500

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The designated pose_file_path does not exist, please check the path and try again")

    def detector(self, img):
        """

        :param img:
        :param threshold:
        :return:
        """
        detect = cv2.FastFeatureDetector_create(threshold=self.threshold, nonmaxSuppression=True)
        point_ref = detect.detect(img)
        return np.array([x.pt for x in point_ref], dtype=np.float32).reshape(-1, 1, 2)

    def get_absolute_scale(self, frame_id):
        """

        :param frame_id:
        :return:
        """
        pose = self.pose[frame_id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[frame_id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))

    def feature_tracking(self, prev_image, curr_image, prevFeatures):
        """

        :param prev_image:
        :type prev_image:
        :param prevFeatures:
        :type prevFeatures:
        :param curr_image:
        :type curr_image:
        :return:
        """
        point_now, status, err = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, prevFeatures,
                                                          None, **self.lk_param)
        # save the good points
        # print(status)
        status = status.reshape(status.shape[0])
        # print(status)
        kp1 = prevFeatures[status == 1]
        kp2 = point_now[status == 1]
        return kp1, kp2

    def process_frame(self, image, traj, color, title, scale_factor=3):
        # cv2.imshow('frame from vo ' + title, image)
        print("Frame number : {} ".format(self.id) + title)
        if self.id == 0:
            self.img_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.id += 1

        elif self.id == 1:
            self.img_2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.prevFeatures = self.detector(self.img_1)
            kp_1, kp_2 = self.feature_tracking(self.img_1, self.img_2, self.prevFeatures)
            E, mask = cv2.findEssentialMat(kp_2, kp_1, self.focal_length,
                                           self.pp, cv2.RANSAC, 0.999, 1.0)
            _, R, t, _ = cv2.recoverPose(E, kp_2, kp_1, self.R,
                                         self.t, self.focal_length, self.pp, mask)
            self.previmage = self.img_2
            self.prevFeatures = kp_2
            self.curr_r = R
            self.curr_t = t

            self.id += 1

        else:
            self.cur_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp_1, kp_2 = self.feature_tracking(self.previmage, self.cur_image, self.prevFeatures)
            E, mask = cv2.findEssentialMat(kp_2, kp_1, self.focal_length,
                                           self.pp, cv2.RANSAC, 0.999, 1.0)
            _, R, t, _ = cv2.recoverPose(E, kp_2, kp_1, self.R,
                                         self.t, self.focal_length, self.pp, mask)
            scale = self.get_absolute_scale(self.id)
            # print(scale)
            if scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0]):
                self.curr_t = self.curr_t - scale * self.curr_r.dot(t)
                self.curr_r = R.dot(self.curr_r)

            if self.prevFeatures.shape[0] < self.kMinNumFeature:
                self.prevFeatures = self.detector(self.previmage)
                self.prevFeatures, kp_2 = self.feature_tracking(self.previmage, self.cur_image, self.prevFeatures)

            self.previmage = self.cur_image
            self.prevFeatures = kp_2

            x, y, z = self.curr_t[0], self.curr_t[1], self.curr_t[2]
            # print(x) #, y, z)
            draw_x = int(-x * scale_factor) + 340  # modify here for different direction
            draw_y = int(z * scale_factor) + 290
            # print(draw_x, draw_y)
            cv2.circle(traj, (draw_x, draw_y), 1, color, 2)

            self.id += 1

    def get_mono_coordinate(self):
        if self.id < 3:
            adj_coord = np.array([[0], [0], [0]])
        else:
            diag = np.array([[-1, 0, 0],
                             [0, -1, 0],
                             [0, 0, -1]])
            adj_coord = np.matmul(diag, self.curr_t)
        return adj_coord.flatten()

    def get_true_coordinates(self):
        """Returns true coordinates of vehicle

        Returns:
            np.array -- Array in format [x, y, z]
        """
        if self.id < 3:
            true_coord = np.array([[0], [0], [0]])
        else:
            pose = self.pose[self.id].strip().split()
            x = float(pose[3])
            y = float(pose[7])
            z = float(pose[11])
            true_coord = np.array([[x], [y], [z]])
        return true_coord.flatten()
