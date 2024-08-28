import os
import json
import cv2
import argparse


def convert_absolute_to_relative(data, image_width, image_height):
    # Iterate over people
    for person in data["people"]:
        # Convert pose keypoints 2D to relative coordinates
        pose_keypoints_2d = person["pose_keypoints_2d"]
        for i in range(0, len(pose_keypoints_2d) - 12, 3):
            width_ratio = 512 / image_width
            height_ratio = 1024 / image_height
            pose_keypoints_2d[i] = pose_keypoints_2d[i] * width_ratio / 512
            pose_keypoints_2d[i + 1] = pose_keypoints_2d[i + 1] * height_ratio / 1024

        # del person["face_keypoints_2d"]
        #
        # del person["hand_left_keypoints_2d"]
        #
        # del person["hand_right_keypoints_2d"]

    return data


def convert_center(data):
    for person in data["people"]:
        pose_keypoints_2d = person["pose_keypoints_2d"]
        center_difference_x = 128 - pose_keypoints_2d[3]
        center_difference_y = 44 - pose_keypoints_2d[4]
        for i in range(0, len(pose_keypoints_2d) - 12, 3):
            if pose_keypoints_2d[i] != 0 and pose_keypoints_2d[i + 1] != 0:
                pose_keypoints_2d[i] = round(pose_keypoints_2d[i] + center_difference_x, 4)
                pose_keypoints_2d[i + 1] = round(pose_keypoints_2d[i + 1] + center_difference_y, 4)
    return data


def remove_confidence(data):
    # Iterate over people
    for person in data["people"]:
        # Remove confidence from pose keypoints 2D
        person["pose_keypoints_2d"] = [num for i, num in enumerate(person["pose_keypoints_2d"]) if (i + 1) % 3 != 0]

    return data


def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return width, height


def save_json_file(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def process_image_json_pairs_absolute(json_folder, result_folder):
    # Ensure train folder exists
    os.makedirs(result_folder, exist_ok=True)

    # Iterate over JSON files
    for json_file in os.listdir(json_folder):
        if json_file.endswith("_keypoints.json"):
            # Read JSON file
            data = read_json_file(os.path.join(json_folder, json_file))

            # Convert absolute coordinates to relative coordinates
            converted_data = convert_center(data)

            # Remove confidence information
            converted_data = remove_confidence(converted_data)

            # 创建 json 文件夹路径
            json_result_folder = os.path.join(result_folder)

            # 如果 json 文件夹不存在，则创建
            if not os.path.exists(json_result_folder):
                os.makedirs(json_result_folder)

            # 构建完整的输出文件路径
            output_file = os.path.join(json_result_folder, json_file)

            # Save converted data to new JSON file
            save_json_file(converted_data, output_file)

            print(f"Converted data saved to {output_file}")


def process_image_json_pairs_relative(image_folder, json_folder, result_folder):
    # Ensure train folder exists
    os.makedirs(result_folder, exist_ok=True)

    # Iterate over JSON files
    for json_file in os.listdir(json_folder):
        if json_file.endswith("_keypoints.json"):
            # Read JSON file
            data = read_json_file(os.path.join(json_folder, json_file))

            # Get corresponding image file name
            image_prefix = json_file.split("_keypoints.json")[0]
            image_file = image_prefix + "_rendered.png"
            image_path = os.path.join(image_folder, image_file)

            # Get image dimensions
            image_width, image_height = get_image_dimensions(image_path)

            # Convert absolute coordinates to relative coordinates
            converted_data = convert_absolute_to_relative(data, image_width, image_height)

            # Remove confidence information
            converted_data = remove_confidence(converted_data)

            # 创建 json 文件夹路径
            json_result_folder = os.path.join(result_folder)

            # 如果 json 文件夹不存在，则创建
            if not os.path.exists(json_result_folder):
                os.makedirs(json_result_folder)

            # 构建完整的输出文件路径
            output_file = os.path.join(json_result_folder, json_file)

            # Save converted data to new JSON file
            save_json_file(converted_data, output_file)

            print(f"Converted data saved to {output_file}")


# Example usage
# process_image_json_pairs_relative("img", "json", "results")


if __name__ == "__main__":
    motions = ['roar', 'samba_dancing', 'meia_lua_de']
    for m in motions:
        source_path = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\json\\seq', m)
        result_path = os.path.join('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\train\\json\\seq', m)
        for folder_name in os.listdir(source_path):
            folder_path = os.path.join(source_path, folder_name)
            if os.path.isdir(folder_path):
                # 构造result路径下的对应文件夹路径
                result_folder_path = os.path.join(result_path, folder_name)
                # 如果result路径下的文件夹不存在，则创建它
                if not os.path.exists(result_folder_path):
                    os.makedirs(result_folder_path)
                process_image_json_pairs_absolute(folder_path, result_folder_path)
    # process_image_json_pairs_absolute('D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\json\\seq\\snatch\\abe',
    #                                   'D:\\pythonProjects\\openpose-skeleton-transformer\\datasets\\train\\json\\seq\\snatch\\abe')
