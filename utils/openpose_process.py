import os
import subprocess

# Set OpenPose installation directory
openpose_install_dir = r"D:\openpose"
motions = ['roar', 'samba_dancing', 'meia_lua_de']
names = ['abe', 'brian', 'drake', 'elizabeth', 'james', 'leonard', 'lewis', 'maynard', 'megan', 'remy', 'sophie',
         'shannon']
# Change to OpenPose installation directory
os.chdir(openpose_install_dir)

# Set image and JSON data root directories
image_root_dir = r"D:\pythonProjects\openpose-skeleton-transformer\datasets\img\seq"
json_root_dir = r"D:\pythonProjects\openpose-skeleton-transformer\datasets\json\seq"

# Iterate over all folders in the image root directory
for motion in motions:
    for name in names:
        folder_name = os.path.join(motion, name)
        image_dir = os.path.join(image_root_dir, folder_name)
        write_images_dir = os.path.join(image_root_dir, motion, 'openpose')
        write_json_dir = os.path.join(json_root_dir, folder_name)
        if not os.path.exists(write_json_dir):
            os.makedirs(write_json_dir)
        # Build command line command
        command = [
            'bin\\OpenPoseDemo.exe',
            '--image_dir', image_dir,
            '--write_images', write_images_dir,
            '--write_json', write_json_dir,
            '--display', '0',
            '--disable_blending',
            '--number_people_max', '1',
            '--model_pose', 'COCO'
        ]

        # Execute command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while processing {folder_name}: {e}")
            print(f"Return code: {e.returncode}")

print("OpenPose processing completed.")
