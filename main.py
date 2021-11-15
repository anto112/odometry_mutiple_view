import numpy as np
import cv2
import os
from Moildev import Moildev
from mono_vo import MonoVisualOdometry

file_path = "../round-dataset/Left/"
pose_path = "../round-dataset/pose/00.txt"
focal_length = 616.8560
camera_param = "Intel-T265_L.json"
moildev = []
vo = MonoVisualOdometry(file_path, pose_path, focal_length, camera_param, int(25))
moildev_a = Moildev(camera_param)
mapx, mapy = moildev_a.getAnypointMaps(0, 0, 3.4)
traj = np.zeros((600, 500, 3), dtype=np.uint8)

color = [(255, 255, 0), (0, 255, 0), (0, 0, 255), (100, 100, 255), (100, 0, 200)]
# position = [(20, 70), (20, 90), (20, 110)]
# position2 = [(20, 520), (20, 540), (20, 560)]
angle = [0, 15, 0, 0, 0, 15, 20, 0, -20, 0]
scale_factor = 3

mapsX = []
mapsY = []
x_list = []
y_list = []
z_list = []
x_ave = 0
y_ave = 0
title = ["Left", "Front", 'Right', "up", "down"]


def create_maps():
    for i in range(5):
        moildev.append(Moildev(camera_param))
    mapx, mapy = moildev[0].getAnypointMaps(angle[0], angle[1], 3.4)
    mapsX.append(mapx)
    mapsY.append(mapy)
    mapx, mapy = moildev[1].getAnypointMaps(angle[2], angle[3], 3.4)
    mapsX.append(mapx)
    mapsY.append(mapy)
    mapx, mapy = moildev[2].getAnypointMaps(angle[4], angle[5], 3.4)
    mapsX.append(mapx)
    mapsY.append(mapy)
    mapx, mapy = moildev[3].getAnypointMaps(angle[6], angle[7], 3.4)
    mapsX.append(mapx)
    mapsY.append(mapy)
    mapx, mapy = moildev[4].getAnypointMaps(angle[8], angle[9], 3.4)
    mapsX.append(mapx)
    mapsY.append(mapy)


def get_ground_truth(pose_file_path, frame_id):
    try:
        with open(pose_file_path) as f:
            pose = f.readlines()
    except Exception as e:
        print(e)
        raise ValueError("The designated pose_file_path does not exist, please check the path and try again")

    pose = pose[frame_id].strip().split()
    x = float(pose[3])
    y = float(pose[7])
    z = float(pose[11])
    return x, y, z


def main_1_view():
    global x_list, y_list, z_list, x_ave, y_ave
    try:
        if not all([".png" in x for x in os.listdir(file_path)]):
            raise ValueError("img_file_path is now correct and does not exclusively png files")
    except Exception as e:
        print(e)
        raise ValueError("The designated img_file_path does not exist, please check the path and try again")

    id = 1
    while id < len([name for name in os.listdir(file_path)]):
        # print("this id from main :{}".format(id))
        cv2.rectangle(traj, (0, 480), (500, 600), (0, 0, 0), -1)
        true_x, true_y, true_z = get_ground_truth(pose_path, id - 1)
        filename = file_path + str(id) + ".png"
        fram = cv2.imread(filename)
        frame = cv2.remap(fram, mapx, mapy, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
        vo.process_frame(frame, traj, (255, 0, 0), "front", scale_factor)
        frame = cv2.resize(frame, (400, 380), interpolation=cv2.INTER_AREA)
        if id > 2:
            x_ave = vo.curr_t[0]
            y_ave = vo.curr_t[2]

        x, y = int(true_x * scale_factor) + 340, int(true_z * scale_factor) + 290
        x_ave, y_ave = int(-x_ave * scale_factor) + 340, int(y_ave * scale_factor) + 290
        cv2.circle(traj, (x, y), 1, (255, 255, 255), 2)
        cv2.circle(traj, (int(x_ave), int(y_ave)), 1, (0, 255, 255), 2)
        cv2.putText(traj, "Multi View Visual Odometry using Moildev", (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))
        cv2.putText(traj, "====================================================", (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 100, 200), int(1))
        cv2.putText(traj, "This color is ground truth", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                    int(0.4))

        cv2.putText(traj, "Writer by: Haryanto", (250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))
        cv2.putText(traj, "Program Language: Python3", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))
        cv2.putText(traj, "Advisor: Chuang-jan Chang", (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))

        text2 = "Angle Left: {},{}, Angle Front: {},{}, Angle Right: {},{}".format(*[str(pt) for pt in angle])
        cv2.putText(traj, text2, (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    int(0.2))
        text = "Groundtruth: x=%.2fm y=%.2fm z=%.2fm" % (true_x, true_y, true_z)
        cv2.putText(traj, text, (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    int(0.3))
        cv2.imshow("Trajectory", traj)

        cv2.imshow(" View", frame)

        k = cv2.waitKey(30)
        if k == ord('q'):
            break
        elif k == ord('p'):
            cv2.waitKey(-1)

        id += 1
    cv2.imwrite("trajectory_1_view_600.png", traj)


def main():
    global x_list, y_list, z_list, x_ave, y_ave
    try:
        if not all([".png" in x for x in os.listdir(file_path)]):
            raise ValueError("img_file_path is now correct and does not exclusively png files")
    except Exception as e:
        print(e)
        raise ValueError("The designated img_file_path does not exist, please check the path and try again")

    new_vo = []
    for i in range(5):
        new_vo.append(MonoVisualOdometry(file_path, pose_path, focal_length, camera_param, int(25)))

    create_maps()

    id = 1
    while id < len([name for name in os.listdir(file_path)]):
        # print("this id from main :{}".format(id))
        cv2.rectangle(traj, (0, 480), (500, 600), (0, 0, 0), -1)
        true_x, true_y, true_z = get_ground_truth(pose_path, id - 1)
        filename = file_path + str(id) + ".png"
        fram = cv2.imread(filename)
        for i in range(3):
            frame = cv2.remap(fram, mapsX[i], mapsY[i], cv2.INTER_CUBIC, cv2.BORDER_CONSTANT)
            # new_vo[i].process_frame(frame, color[i], traj, title[i], position[i], position2[i], scale_factor)
            new_vo[i].process_frame(frame, traj, color[i], title[i], scale_factor)
            if id > 2:
                x_list.append(new_vo[i].curr_t[0])
                y_list.append(new_vo[i].curr_t[1])
                z_list.append(new_vo[i].curr_t[2])

            frame = cv2.resize(frame, (400, 380), interpolation=cv2.INTER_AREA)
            cv2.imshow(title[i] + " View", frame)

            # print("this is true z {}".format(abs(true_z + 1)))
            # print(np.average(z_list))
        if id > 2:
            x_ave = np.average(x_list)
            y_ave = np.average(z_list)

        x_list = []
        y_list = []
        z_list = []

        # print("this is trux - 20%: {}".format(abs(true_x - ((true_x * 30) / 100))))
        # print("this is trux - 20%: {}".format(abs(true_x + ((true_x * 30) / 100))))
        x, y = int(true_x * scale_factor) + 340, int(true_z * scale_factor) + 290
        x_ave, y_ave = int(-x_ave * scale_factor) + 340, int(y_ave * scale_factor) + 290
        cv2.circle(traj, (x, y), 1, (255, 255, 255), 2)
        cv2.circle(traj, (int(x_ave), int(y_ave)), 1, (0, 255, 255), 2)
        cv2.putText(traj, "Multi View Visual Odometry using Moildev", (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))
        cv2.putText(traj, "====================================================", (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 100, 200), int(1))
        cv2.putText(traj, "This color is ground truth", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                    int(0.4))

        cv2.putText(traj, "Writer by: Haryanto", (250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))
        cv2.putText(traj, "Program Language: Python3", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))
        cv2.putText(traj, "Advisor: Chuang-jan Chang", (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 100, 200), int(1))

        text2 = "Angle Left: {},{}, Angle Front: {},{}, Angle Right: {},{}".format(*[str(pt) for pt in angle])
        cv2.putText(traj, text2, (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    int(0.2))
        text = "Groundtruth: x=%.2fm y=%.2fm z=%.2fm" % (true_x, true_y, true_z)
        cv2.putText(traj, text, (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    int(0.3))
        cv2.imshow("Trajectory", traj)

        k = cv2.waitKey(30)
        if k == ord('q'):
            break
        elif k == ord('p'):
            cv2.waitKey(-1)

        id += 1
    cv2.imwrite("trajectory.png", traj)


if __name__ == "__main__":
    # print(len([name for name in os.listdir(file_path)]))
    # main_1_view()
    main()
