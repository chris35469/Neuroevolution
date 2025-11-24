import os
import shutil

def clear_folder(folder_path):
    """
    Clear everything from a folder.
    
    Args:
        folder_path: Path to the folder to clear
    """
    if os.path.exists(folder_path):
        # Remove all contents
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        print(f'Successfully cleared all contents from {folder_path}/')
    else:
        print(f'Folder {folder_path}/ does not exist.')

def clear_all():
    """
    Clear everything from folders.
    """
    print('Clearing folders...')
    print('=' * 60)
    clear_folder('notrain_ycommand')
    clear_folder('notrain_ycontinuous')
    clear_folder('train_ycommand')
    clear_folder('train_ycontinuous')
    print('=' * 60)
    print('Done!')

if __name__ == '__main__':
    clear_all()

