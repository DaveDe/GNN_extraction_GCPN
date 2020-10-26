import torch

from torch_geometric.utils import dense_to_sparse

#Convert a batch of dense adjacency matrices to edge_indices (sparse block-diagonal matrix)
def states_to_edge_indicies(states):

	#Each entry assigns a node to a graph
	batch = []

	dense_adj_lst = []

	for i in range(states.shape[0]):

		dense_adj_lst.append(states[i])
		batch.extend([i for x in range(states[i].shape[0])])

	block_diag = torch.block_diag(*dense_adj_lst)

	#Returned batch is wrong due to self loops. So we get batch ourselves
	edge_indices, _ = dense_to_sparse(block_diag)

	return edge_indices, torch.Tensor(batch).long()


def plot_graph(adj, i):
	G = nx.from_numpy_matrix(adj)
	G.remove_nodes_from(list(nx.isolates(G)))
	pos = nx.spring_layout(G)
	f1 = plt.figure(i)
	nx.draw_networkx_edges(G, pos, edge_color='r')
	nx.draw_networkx_nodes(G, pos, node_color='b', node_size=4)
	plt.title('Graph'+str(i))
	f1.show()
	graph_name = "graph_images/graph_"+str(i)+".png"
	plt.savefig(graph_name)

