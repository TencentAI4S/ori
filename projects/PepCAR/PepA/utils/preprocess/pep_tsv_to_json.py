import json

def tsv_to_json(input_file, output_file):
    data = {}
    with open(input_file, "r") as f:
        for line in f:
            id, aa, smile = line.strip().split("\t")
            data[id] = {
                "aa": aa,
                "smile": smile
            }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def main():
    input_file = "pep_seq_smiles.tsv"
    output_file = "pep_seq_smiles.json"
    tsv_to_json(input_file, output_file)

if __name__ == "__main__":
    main()
