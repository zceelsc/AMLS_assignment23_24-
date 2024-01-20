import subprocess
import os

locationa = r'C:\Users\user\AMLS_assignment23_24-\A'  # Change this accordingly
locationb = r'C:\Users\user\AMLS_assignment23_24-\B'  # Change this accordingly
dataset_location = r'C:\Users\user\AMLS_assignment23_24-\Datasets'  # Change this to the desired dataset location
model_locationa = os.path.join(locationa, 'A_trained_model.pth')
model_locationb = os.path.join(locationb, 'B_trained_model.pth')

def run_A_py():
    script_path = os.path.join(locationa, 'A.py')
    print(f"Attempting to run: {script_path}")
    
    # Change the current working directory to the location of A.py
    os.chdir(locationa)

    try:
        subprocess.run(['python', script_path, dataset_location, model_locationa], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running A.py: {e}")

def run_B_py():
    script_path = os.path.join(locationb, 'B.py')
    print(f"Attempting to run: {script_path}")
    
    # Change the current working directory to the location of B.py
    os.chdir(locationb)

    try:
        subprocess.run(['python', script_path, dataset_location, model_locationb], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running B.py: {e}")

option = input('A or B: ')  # input A or B for binary or multi-class

# Call the function to run the selected script
if option.lower() == 'a':
    run_A_py()
elif option.lower() == 'b':
    run_B_py()
else:
    print('Input error')
