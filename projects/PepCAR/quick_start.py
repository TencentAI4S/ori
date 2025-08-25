import subprocess
import sys
import os
def run_command(command):
    """Run a shell command and print the output in real-time."""
    print(f"Running command: {command}")

    os.system(command)

def main():
    print("Welcome to the automation script for in vitro and in vivo prediciton, as well as peptide evolution!")
    
    while True:
        print("\nPlease choose an option:")
        print("1. Run PepC in vitro prediction for PDBbind")
        print("2. Run PepC in vitro prediction for single pair")
        print("3. Run PepC in vitro prediction for antigen-HLA")
        print("4. Run PepA in vivo prediction (first-stage)")
        print("5. Run PepA in vivo prediction (second-stage)")
        print("6. Run PepR peptide evolution")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            run_command("cd PepC && python predict.py --task pdbbind")
            print("Results were saved in PepC/output/pdbbind.tsv")
            break
        elif choice == '2':
            run_command("cd PepC && python predict.py --task single")
            break
        elif choice == '3':
            run_command("cd PepC && python predict.py --task pmhc")
            print("Results were saved in PepC/output/pmhc.tsv")
            break
        elif choice == '4':
            run_command("cd PepA && python predict.py --task admet")
            break
        elif choice == '5':
            run_command("cd PepA && python predict.py --task stability")
            break
        elif choice == '6':
            run_command("cd PepR && sh run.sh")
            break
        elif choice == '7':
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()