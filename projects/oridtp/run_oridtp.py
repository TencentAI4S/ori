import subprocess
import sys
import os
def run_command(command):
    """Run a shell command and print the output in real-time."""
    print(f"Running command: {command}")

    os.system(command)

def main():
    print("Welcome to the ORIDTP automation script!")
    
    while True:
        print("\nPlease choose an option:")
        print("1. Run PepDif generation (Generic model)")
        print("2. Run PepDif generation (GLP1R&GCGR enhanced model)")
        print("3. Run PepDRL optimization")
        print("4. Run single prediction in PepDAF")
        print("5. Run batch prediction in PepDAF")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            run_command("cd PepDif/scripts && sh run_decode.sh")
            break
        elif choice == '2':
            run_command("cd PepDif/scripts && sh run_decode_glp1rgcgr.sh")
            break
        elif choice == '3':
            run_command("cd PepDRL && sh run.sh")
            break
        elif choice == '4':
            run_command("cd PepDAF && python predict.py --task single")
            break
        elif choice == '5':
            run_command("cd PepDAF/utils/preprocess && sh start.sh")
            run_command("cd PepDAF && python predict.py --task batch")
            break
        elif choice == '6':
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()