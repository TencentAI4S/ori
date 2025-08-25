from experiment import FPAExperiment
import argparse
from parsing import add_train_args

def initialize_experiment(args):
    """Initialize the FPAExperiment with the provided arguments."""
    return FPAExperiment(
        task=args.task,
        split_method='test',
        contact_cutoff=args.contact_cutoff,
        num_rbf=args.num_rbf,
        prot_gcn_dims=args.prot_gcn_dims,
        prot_fc_dims=args.prot_fc_dims,
        pep_gcn_dims=args.smiles_gcn_dims,
        pep_fc_dims=args.smiles_fc_dims,
        smiles_gcn_dims=args.smiles_gcn_dims,
        smiles_fc_dims=args.smiles_fc_dims,
        mlp_dims=args.mlp_dims,
        mlp_dropout=args.mlp_dropout,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

def get_checkpoints(base_path, num_folds=5):
    """Generate a list of checkpoint file paths."""
    return [f"{base_path}/fold_{i}.pt" for i in range(1, num_folds + 1)]

def main():
    parser = argparse.ArgumentParser(description='Run Propedia experiment')
    
    # Add training arguments
    add_train_args(parser)
    
    # Add task argument
    parser.add_argument('--task', type=str, choices=['admet', 'stability'], default='stability',
                        help="Specify the task to run the first stage (ADMET) or the second stage (stability) prediction.")

    args = parser.parse_args()

    # Initialize the experiment
    exp = initialize_experiment(args)

    if args.task == 'admet':
        properties = ['ames', 'bbb', 'cyp', 'logd', 'microsomal_cl', 'ppb', 'solubility']
        seq = 'FSGTVTTAGLLF'
        print(f'Stage 1 prediction for {seq}')
        for i in range(len(properties)):
            path = f'../model_weights/PepA_first_stage/{properties[i]}'
            checkpoints = get_checkpoints(path)
            r = exp.predict_ADMET(checkpoints, peptide_seq=seq, property_name=properties[i])
            print(f"ADMET Property [{properties[i]}]: {r}")
    
    elif args.task == 'stability':
        properties = ['invivo', 'invitro']
        seq = 'FSGTVTTAGLLF'
        print(f'Stage 2 prediction for {seq}')
        for i in range(len(properties)):
            path = f'../model_weights/PepA_second_stage/{properties[i]}'
            checkpoints = get_checkpoints(path)
            r = exp.predict_stability(checkpoints, peptide_seq=seq, property_name=properties[i])
            print(f"Stability [{properties[i]}-based]: {r}")

if __name__ == '__main__':
    main()