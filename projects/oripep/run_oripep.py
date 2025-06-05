import subprocess
import sys
import os
def run_command(command):
    """Run a shell command and print the output in real-time."""
    print(f"Running command: {command}")

    os.system(command)

def main():
    print("Welcome to the ORIPEP automation script!")
    
    while True:
        print("\nPlease choose an option:")
        print("1. Run PepGPT generation")
        print("2. Run PepRL optimization")
        print("3. Run single prediction in PepAF")
        print("4. Run batch prediction in PepAF")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            run_command("cd PepGPT && python sample.py")
            break
        elif choice == '2':
            run_command("cd PepRL && sh run.sh")
            break
        elif choice == '3':
            run_command("cd PepAF && python predict.py --task single")
            break
        elif choice == '4':
            run_command("cd PepAF/utils/preprocess && sh start.sh")
            run_command("cd PepAF && python predict.py --task batch")
            break
        elif choice == '5':
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()