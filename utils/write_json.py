import json
import os

def write_to_json(points, filename='pose_keypoints_2d.json'):
    folder_name = os.path.dirname(filename)
    # 检查文件夹是否存在，如果不存在则创建
    if folder_name:
        os.makedirs(folder_name, exist_ok=True)
    points = points.reshape(-1).tolist()
    # points = [0 if point < 1 else point for point in points]
    # 创建 JSON 数据
    data = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": points,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
        ]
    }

    # 将数据写入 JSON 文件
    with open(filename, 'w') as f:
        json.dump(data, f)
