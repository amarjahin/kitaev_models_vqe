# kitaev models VQE
VQE implementation for the Kitaev model using Majorana fermions.

In the case of zero magnetic field it's more computationally efficient to work in a fixed gauge subspace. 
The vqe_u_set.py file is for running this case of zero magnetic field. 

In the case of non-zero magnetic field is dealt with (not perfectly) in the vqe_setup.py file.

# Things to add 
1. Try modifying the ansatz to include exp(i \theta bbcc) terms. 