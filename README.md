# kitaev models VQE
VQE implementation for the Kitaev model using Majorana fermions.
To run the code you need to run the vqe_setup.py file

# Things to get fixed
1. The way the fermionic Hamiltonian gets defined relies on the fact that the graph is bipartite. This causes problems with the 4-8-8 model since it's not bipartite. 
