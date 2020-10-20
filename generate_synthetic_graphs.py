import pickle

from generate_graphs.generate_synthetic_graphs import gen_synthetic_graph1
from generate_graphs.generate_synthetic_graphs import gen_synthetic_graph2
from generate_graphs.generate_synthetic_graphs import gen_synthetic_graph3


#GENERATE TREE/GRID/STAR TRAINING GRAPHS
num_graphs = 100
adj_list, edge_indicies_list, features_list, labels_list, class_distribution = gen_synthetic_graph1(num_graphs)
print(class_distribution)


with open('save/graphs/tree_grid_star_train.pkl',"wb") as f:
	pickle.dump([adj_list, edge_indicies_list, features_list, labels_list, class_distribution], f)

#GENERATE TREE/GRID/STAR EVAL GRAPHS
num_graphs = 300
adj_list, edge_indicies_list, features_list, labels_list, class_distribution = gen_synthetic_graph1(num_graphs)

with open('save/graphs/tree_grid_star_eval.pkl',"wb") as f:
	pickle.dump([adj_list, edge_indicies_list, features_list, labels_list, class_distribution], f)





#GENERATE TREE/CYCLE/STAR TRAINING GRAPHS
"""num_graphs = 100
adj_list, edge_indicies_list, features_list, labels_list, class_distribution = gen_synthetic_graph2(num_graphs)

with open('save/graphs/tree_cycle_star_train.pkl',"wb") as f:
	pickle.dump([adj_list, edge_indicies_list, features_list, labels_list, class_distribution], f)

#GENERATE TREE/CYCLE/STAR EVAL GRAPHS
num_graphs = 500
adj_list, edge_indicies_list, features_list, labels_list, class_distribution = gen_synthetic_graph2(num_graphs)

with open('save/graphs/tree_cycle_star_eval.pkl',"wb") as f:
	pickle.dump([adj_list, edge_indicies_list, features_list, labels_list, class_distribution], f)
"""


"""
#GENERATE TREE/CYCLE/BA TRAINING GRAPHS
num_graphs = 100
adj_list, edge_indicies_list, features_list, labels_list, class_distribution = gen_synthetic_graph3(num_graphs)
#utils.plot_graph(adj_list[0].numpy(),0)
print(class_distribution)


with open('save/graphs/ba_cycle_star_train.pkl',"wb") as f:
	pickle.dump([adj_list, edge_indicies_list, features_list, labels_list, class_distribution], f)

#GENERATE TREE/CYCLE/BA EVAL GRAPHS
num_graphs = 500
adj_list, edge_indicies_list, features_list, labels_list, class_distribution = gen_synthetic_graph3(num_graphs)

with open('save/graphs/ba_cycle_star_eval.pkl',"wb") as f:
	pickle.dump([adj_list, edge_indicies_list, features_list, labels_list, class_distribution], f)

"""