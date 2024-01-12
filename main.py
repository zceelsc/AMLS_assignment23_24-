import subprocess
location=r'C:\Users\user\AMLS_assignment23_24-\Datasets'  #change this accordingly
def run_A_py():
    try:
        subprocess.run(['python', 'A.py', location], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running a.py: {e}")

def run_B_py():
    try:
        subprocess.run(['python', 'B.py', location], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running B.py: {e}")

option = input('A or B: ')   #input A or B for binary or multi-class
# Call the function to run a.py
if option=='A' or option =='a':
    run_A_py()
elif option=='B' or option =='b':
    run_B_py()
else:
    print('input error')
