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
    parser.add_argument('--task', type=str, choices=['pdbbind', 'single', 'pmhc'], default='single',
                        help="Specify the task to run: 'batch' for batch prediction or 'single' for single prediction.")

    args = parser.parse_args()

    # Initialize the experiment
    exp = initialize_experiment(args)
    if args.task == 'pmhc':
        path = '../model_weights/PepC_pmhc'
        checkpoints = get_checkpoints(path)
    else:
        path = '../model_weights/PepC'
        checkpoints = get_checkpoints(path)

    if args.task == 'pdbbind':
        test_results = exp.predict_pdbbind(checkpoints=checkpoints, save_prediction=True, save_df_name=f'{args.task}.tsv')
    
    elif args.task == 'single':
        exp.predict_single(checkpoints, protein_id='5yqz_R', peptide_seq='HSQGTFTSDYSKYLDSERAQEFVQWLENE')

    elif args.task == 'pmhc':
        test_results = exp.predict_pmhc(checkpoints=checkpoints, save_prediction=True, save_df_name=f'{args.task}.tsv')
    
    else:
        raise ValueError(f"Unknown task '{args.task}'. Please specify 'batch' or 'single'.")

if __name__ == '__main__':
    main()