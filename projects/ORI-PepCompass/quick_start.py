import subprocess
import sys
import os
def run_command(command):
    """Run a shell command and print the output in real-time."""
    print(f"Running command: {command}")

    os.system(command)

def main():
    print("Welcome to the automation script for ORI-PepCompass: bioactivity prediction, developability assessment, and peptide optimization!")
    
    while True:
        print("\nPlease choose an option:")
        print("1. Run PepB bioactivity prediction for PDBbind")
        print("2. Run PepB bioactivity prediction for single pair")
        print("3. Run PepB bioactivity prediction for antigen-HLA")
        print("4. Run PepD unified 9-property developability assessment (single peptide)")
        print("5. Run PepD unified 9-property developability assessment (batch)")
        print("6. Run PepR peptide evolution")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            run_command("cd PepB && python predict.py --task pdbbind")
            print("Results were saved in PepB/output/pdbbind.tsv")
            break
        elif choice == '2':
            run_command("cd PepB && python predict.py --task single")
            break
        elif choice == '3':
            run_command("cd PepB && python predict.py --task pmhc")
            print("Results were saved in PepB/output/pmhc.tsv")
            break
        elif choice == '4':
            run_command("cd PepD && python predict.py --sequence GSPLRYQWTIDYKGR --gpu 0")
            break
        elif choice == '5':
            run_command("cd PepD && python predict.py --input peptides.txt --output scored.tsv --gpu 0")
            break
        elif choice == '6':
            run_command("cd PepR && sh run.sh")
            break
        elif choice == '7':
            print("Exiting the script.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

if __name__ == "__main__":
    main()