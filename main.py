import subprocess

def run_A_py():
    try:
        subprocess.run(['python', 'A.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running a.py: {e}")


def run_B_py():
    try:
        subprocess.run(['python', 'B.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running B.py: {e}")



option = input('first task or second task: ')
# Call the function to run a.py
if option=='A' or option =='a':
    run_A_py()
elif option=='B' or option =='b':
    run_B_py()
else:
    print('input error')
