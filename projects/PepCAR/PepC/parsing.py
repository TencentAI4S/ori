def add_train_args(parser):
    # Data representation parameters
    parser.add_argument('--contact_cutoff', type=float, default=8.,
        help='cutoff of C-alpha distance to define protein contact graph')
    parser.add_argument('--num_pos_emb', type=int, default=16,
        help='number of positional embeddings')
    parser.add_argument('--num_rbf', type=int, default=16,
        help='number of RBF kernels')
    # Protein model parameters
    parser.add_argument('--prot_gcn_dims', type=int, nargs='+', default=[128, 256, 256],
        help='protein GCN layers dimensions')
    parser.add_argument('--prot_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='protein FC layers dimensions')
    # Smiles model parameters
    parser.add_argument('--smiles_gcn_dims', type=int, nargs='+', default=[128, 256, 256],
        help='smiles GVP hidden layers dimensions')
    parser.add_argument('--smiles_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='smiles FC layers dimensions')

    # Top model parameters
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[1024, 512],
        help='top MLP layers dimensions')
    parser.add_argument('--mlp_dropout', type=float, default=0.25,
        help='dropout rate in top MLP')
    parser.add_argument('--n_ensembles', type=int, default=5,
        help='number of ensembles')
    parser.add_argument('--batch_size', type=int, default=32,
        help='batch size')
    parser.add_argument('--output_dir', action='store', default='./output', help='output folder')