import torch
import nflows
from nflows import flows
from nflows import transforms
from nflows import distributions


def permutation_layer(dim, permutation='1-cycle'):
    if permutation:
        if 'cycle' in permutation:
            n=int(permutation.split('-')[0]) 
            if n<dim:
                p=list(range(dim)[-n:])+list(range(dim)[:-n]) # 3-cyclic : [0,1,2,3,4,5,6,7,8,9] -> [7,8,9,0,1,2,3,4,5,6]    
            else:
                raise ValueError('n-cycle must be a positive integer smaller than dim')
        elif permutation=='inverse':
            p=list(range(dim))
            p.reverse()
        else:
            raise ValueError('wrong permutation arg. Use [n]-cycle or inverse')
    else:
        p=list(range(dim))
    return torch.tensor(p)

def MAF_Affine( dim=3,             
                hidden_dims=64,
                context_dim=None,
                num_flows=5,
                num_blocks=5,
                num_bins=10,
                permutation='1-cycle',
                use_residual_blocks=True,
                random_mask=False,
                coupling_mask=None,
                use_batch_norm=False,
                activation=F.leaky_relu,
                dropout_probability=0.0,
                device='cpu'
                ):
    
    list_transforms=[]

    for _ in range(num_flows):  
        
        flow = nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_dims,
                context_features=context_dim,
                num_blocks=num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=random_mask,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm)
        perm=nflows.transforms.permutations.Permutation(permutation_layer(dim=dim, permutation=permutation))
        list_transforms.append(flow)
        list_transforms.append(perm)
    transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device) 
    base_dist = nflows.distributions.normal.StandardNormal(shape=[dim])
    return nflows.flows.base.Flow(transform, base_dist).to(device)



def MAF_RQS(flow_model='MAF_rational_quadratic',
                    dim=3,             
                    hidden_dims=64,
                    context_dim=None,
                    num_flows=5,
                    num_blocks=5,
                    num_bins=10,
                    permutation='1-cycle',
                    use_residual_blocks=True,
                    random_mask=False,
                    coupling_mask=None,
                    use_batch_norm=False,
                    activation=F.leaky_relu,
                    dropout_probability=0.0,
                    device='cpu'
                    ):
    
    list_transforms=[]

    for _ in range(num_flows):  
        use_residual_blocks=None
        flow = nflows.transforms.autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_dims,
                context_features=context_dim,
                num_blocks=num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=random_mask,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                num_bins=num_bins,
                tails='linear',
                tail_bound=10
                )
        perm = nflows.transforms.permutations.Permutation(permutation_layer(dim=dim, permutation=permutation))
        list_transforms.append(flow)
        list_transforms.append(perm)
    transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device) 
    base_dist = nflows.distributions.normal.StandardNormal(shape=[dim])
    return nflows.flows.base.Flow(transform, base_dist).to(device)







def normalizing_flow(flow_model='MAF_rational_quadratic',
                    dim=3,             
                    hidden_dims=64,
                    context_dim=None,
                    num_flows=5,
                    num_blocks=5,
                    num_bins=10,
                    permutation='1-cycle',
                    use_residual_blocks=True,
                    random_mask=False,
                    coupling_mask=None,
                    use_batch_norm=False,
                    activation=F.leaky_relu,
                    dropout_probability=0.0,
                    device='cpu'
                    ):
    
    list_transforms=[]
    for _ in range(num_flows):  
        
        if flow_model=='MAF_affine':
            flow=nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_dims,
                context_features=context_dim,
                num_blocks=num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=random_mask,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm
                    )
        elif flow_model=='MAF_rational_quadratic':
            use_residual_blocks=None
            flow=nflows.transforms.autoregressive.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_dims,
                context_features=context_dim,
                num_blocks=num_blocks,
                use_residual_blocks=use_residual_blocks,
                random_mask=random_mask,
                activation=activation,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
                num_bins=num_bins,
                tails='linear',
                tail_bound=10
                )
        elif flow_model=='coupling_rational_quadratic':

            mask = torch.ones(dim)
            if coupling_mask=='checkerboard': mask[::2]=-1
            elif coupling_mask=='mid-split': mask[int(dim/2):]=-1  # 2006.08545

            def resnet(in_features, out_features):
                return nflows.nn.nets.ResidualNet(
                    in_features,
                    out_features,
                    context_features=context_dim,
                    hidden_features=hidden_dims,
                    num_blocks=num_blocks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            flow=nflows.transforms.coupling.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=resnet,
                num_bins=num_bins,
                tails='linear',
                tail_bound=10
                )
            
        perm=nflows.transforms.permutations.Permutation(permutation_layer(dim=dim, permutation=permutation))
        list_transforms.append(flow)
        list_transforms.append(perm)
        
    transform = nflows.transforms.base.CompositeTransform(list_transforms).to(device) 
    base_dist = nflows.distributions.normal.StandardNormal(shape=[dim])
    return nflows.flows.base.Flow(transform, base_dist).to(device)
