#!/bin/bash

echo "Starting the process..."

echo "Creating DataFrame..."
python create_df.py
echo "DataFrame created successfully."

echo "Adding SMILES data..."
python add_smiles.py
echo "SMILES data added successfully."

echo "Converting peptide TSV to JSON..."
python pep_tsv_to_json.py
echo "Conversion to JSON completed successfully."

echo "Converting peptides to FASTA format..."
python pep_to_fasta.py
echo "Conversion to FASTA format completed successfully."

echo "Creating directory for IDs..."
mkdir -p ids
echo "Directory 'ids' created."

CURRENT_PATH=$(pwd)
echo "Current path is: $CURRENT_PATH"

echo "Changing directory to ../iupred..."
cd ../iupred || { echo "Failed to change directory to ../iupred"; exit 1; }

echo "Calculating matrix..."
python calculate_matrix.py "$CURRENT_PATH/fasta" "$CURRENT_PATH/ids"
echo "Matrix calculation completed successfully."

echo "Changing back to the original directory..."
cd "$CURRENT_PATH" || { echo "Failed to change back to the original directory"; exit 1; }

echo "Merging results..."
python merge.py
echo "Merging completed successfully."

echo "Extracting peptide embeddings..."
python extract_pep_esm.py
echo "Peptide embeddings extracted successfully."

echo "Extracting protein embeddings..."
python extract_pro_esm.py
echo "Protein embeddings extracted successfully."

echo "Process completed successfully."