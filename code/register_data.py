import os
import subprocess
import argparse

def main(main_directory):
    for folder in os.listdir(main_directory):
        if not folder.startswith('sub-'):
            continue

        folder_path = os.path.join(main_directory, folder)

        t2_file = None
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('_T2w.nii.gz'):
                    t2_file = os.path.join(root, file)
                    break

        if t2_file is not None:
            subject_id = folder.split('_')[0]
            session_id = folder.split('_')[1]

            save_path = os.path.join(folder_path, f'{subject_id}{session_id}_T2w_brain_affine.nii.gz')

            cmd = f"python Fetal/SurfaceExtraction/code/register.py --move_path='{t2_file}' --save_path='{save_path}'"
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                print(f"Registration successful for {folder}")
            else:
                print(f"Registration failed for {folder}. Error output:\n{result.stderr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register data for fetal surface extraction")
    parser.add_argument("--path", required=True, help="Path to the dataset directory.")
    args = parser.parse_args()

    main(args.main_directory)
