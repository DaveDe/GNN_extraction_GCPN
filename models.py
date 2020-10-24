import random

import torch
import torch.nn.functional as F
import torch.autograd as autograd 
import torch.nn as nn

from torch.distributions import Categorical

#from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import GCNConv, GINConv, SAGPooling, global_mean_pool

import utils

class blackbox_synthetic_GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(blackbox_synthetic_GIN, self).__init__()

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)))

        self.lin = nn.Linear(64, num_classes)

    def forward(self, features, edge_indicies):

        x = F.relu(self.conv1(features, edge_indicies))
        x = F.relu(self.conv2(x, edge_indicies))
        x = F.relu(self.conv3(x, edge_indicies))
        x = self.lin(x)

        return x
"""
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, embedding_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
"""


class Critic(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        super(Critic, self).__init__()

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size)))

        self.lin = nn.Linear(embedding_size, 1)

        self.num_features = num_features

    def forward(self, states):

        edge_index, batch = utils.states_to_edge_indicies(states)

        num_nodes = batch.shape[0]

        features = torch.ones((num_nodes, self.num_features))

        x = F.relu(self.conv1(features, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

class Actor(torch.nn.Module):
    def __init__(self, num_features, embedding_size):
        super(Actor, self).__init__()

        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_features, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, embedding_size)))

        #Feed mlp_A embeddings of all non-scaffold nodes
        #Output unormalized distribution over non-scaffold nodes
        self.mlp_A = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        #Feed mlp_B embeddings of all nodes, including scaffold nodes.
        #These embeddings are concatenated with embedding of node selected from mlp_A
        #Output unormalized distribution over all nodes
        self.mlp_B = nn.Sequential(
            nn.Linear(embedding_size*2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        #Feed mlp_C a graph embedding, which is AGG() of all node embeddings
        #Output unormalized distribution over {dont_stop, stop} actions.
        self.mlp_C = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        self.num_features = num_features

    #Feed states through Actor, and sample an action.
    #Return the hierarchical action distributions and the sample
    def forward(self, states):

        num_graphs = states.shape[0]
        num_nodes_per_graph = states[0].shape[0]

        #Compute node embeddings from given states
        edge_index, batch = utils.states_to_edge_indicies(states)
        num_total_nodes = batch.shape[0]

        features = torch.ones((num_total_nodes, self.num_features))

        x = F.relu(self.conv1(features, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        #Get global non-scaffold node indices
        non_scaffold_indicies = []
        non_scaffold_indicies_flattened = []

        all_node_indices = [] #all_node_indices includes indices of scaffold nodes
        all_node_indices_flattened = [] 
        offset = 0

        for state in states:
            node_indices = (state.sum(dim=0) > 0).nonzero().numpy().flatten().tolist()
            node_indices = [i+offset for i in node_indices]
            offset += num_nodes_per_graph

            non_scaffold_indicies.append(node_indices[:-1])
            non_scaffold_indicies_flattened.extend(node_indices[:-1]) #Last node is scaffold

            all_node_indices.append(node_indices)
            all_node_indices_flattened.extend(node_indices)

        #print("Non-scaffold Global indices:", non_scaffold_indicies)

        #Feed non-scaffold node embeddings to mlp_A
        non_scaffold_node_embeddings = x[non_scaffold_indicies_flattened]
        first_node_logits = self.mlp_A(non_scaffold_node_embeddings)
        #print("First node logits:", first_node_logits)

        #For each graph, save distribution over non-scaffold nodes
        i = 0
        probs_A = []

        #Determine the number of nodes in graph with most connected nodes.
        dist_size = max([len(l) for l in non_scaffold_indicies])
        for nodes in non_scaffold_indicies:

            logits = first_node_logits[i:i+len(nodes)]
            logits = torch.reshape(logits, (1, -1)).squeeze(0)
            probs = F.softmax(logits)

            #Add 0's to distribution if this graph has fewer nodes than another
            if(probs.shape[0] < dist_size):

                extended_probs = torch.zeros(dist_size)
                extended_probs[:probs.shape[0]] = probs
                probs_A.append(extended_probs)

            else:

                probs_A.append(probs)

            i += len(nodes)

        probs_A = torch.stack(probs_A, dim=0)

        #print("probs_A:", probs_A)

        dist_A = Categorical(probs_A)

        #print("dist_A", dist_A)
        

        #Sample first nodes
        first_selected_nodes = dist_A.sample()
        #print("First Selected Nodes:", first_selected_nodes)


        #Get global index of selected nodes
        batch_offset = torch.Tensor([g*num_nodes_per_graph for g in range(num_graphs)])
        first_selected_nodes_global = (first_selected_nodes + batch_offset).long()
        #print("Batch offset:", batch_offset)
        #print("first selected nodes global:", first_selected_nodes_global)



        #For second node selection we include scaffold nodes, and concat embeddings with first selected
        #node embedding
        first_node_embeddings = x[first_selected_nodes_global]
        #print("First node embeddings shape:", first_node_embeddings.shape)

        #Repeat first selected node embedding for each node in the graph
        first_node_embeddings = first_node_embeddings.repeat_interleave(num_nodes_per_graph, dim=0)
        #print("Repeated node embeddings shape:", first_node_embeddings.shape)

        concat_node_embeddings = x.clone()
        concat_node_embeddings = torch.cat((concat_node_embeddings, first_node_embeddings), 1)

        #Only consider connected and scaffold nodes
        #print("Indices of non-scaffold + scaffold nodes:", all_node_indices)
        concat_node_embeddings = concat_node_embeddings[all_node_indices_flattened]
        #print("Concat node embeddings shape:", concat_node_embeddings.shape)

        second_node_logits = self.mlp_B(concat_node_embeddings)

        #For each graph, save distribution over scaffold + non-scaffold nodes
        i = 0
        probs_B = []

        #Determine the number of nodes in graph with most connected nodes.
        dist_size = max([len(l) for l in all_node_indices])

        #print("dist_size")

        for nodes in all_node_indices:

            logits = second_node_logits[i:i+len(nodes)]
            logits = torch.reshape(logits, (1, -1)).squeeze(0)
            probs = F.softmax(logits)

            #Add 0's to distribution if this graph has fewer nodes than another
            if(probs.shape[0] < dist_size):

                extended_probs = torch.zeros(dist_size)
                extended_probs[:probs.shape[0]] = probs
                probs_B.append(extended_probs)

            else:

                probs_B.append(probs)

            i += len(nodes)

        probs_B = torch.stack(probs_B, dim=0)

        #print("probs_B:", probs_B)

        dist_B = Categorical(probs_B)

        #print("dist_B:", dist_B)

        second_selected_nodes = dist_B.sample()
        #print("Second Selected Nodes:", second_selected_nodes)



        #Compute graph embeddings to determine whether to end graph construction
        #Only use embeddings of connected and scaffold nodes
        relevant_node_embeddings = x[all_node_indices_flattened]
        batch = []
        for i, nodes in enumerate(all_node_indices):
            batch.extend([i for n in range(len(nodes))])

        #print("relevant node embeddings:", relevant_node_embeddings.shape)
        #print("Batch:", batch)

        graph_embeddings = global_mean_pool(relevant_node_embeddings, torch.Tensor(batch).long())
        #print("Graph embeddings shape:", graph_embeddings.shape)

        end_construction_logits = self.mlp_C(graph_embeddings)
        end_construction_probs = F.softmax(end_construction_logits, dim=1)

        dist_C = Categorical(end_construction_probs)
        isEnd = dist_C.sample()


        #Return distribution per each action component
        #Also return a sampled action, where each column is action component

        action = torch.cat((first_selected_nodes.unsqueeze(0),
                            second_selected_nodes.unsqueeze(0),
                            isEnd.unsqueeze(0)))

        action = torch.transpose(action, 0, 1)

        #print("Action:", action)

        return (dist_A, dist_B, dist_C), action



class ActorCritic(torch.nn.Module):
    def __init__(self, num_features, embedding_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = Critic(num_features, embedding_size)
        self.actor = Actor(num_features, embedding_size)
                
    def forward(self, x):

        value = self.critic(x)
        distributions, samples  = self.actor(x)

        return distributions, samples, value













"""
class DQN(torch.nn.Module):
    def __init__(self, num_features, num_nodes, min_edges, hidden_units=64):
        super(DQN, self).__init__()

        self.conv1 = GINConv(Seq(Lin(num_features, hidden_units), ReLU(), Lin(hidden_units, hidden_units)))
        #self.pool1 = SAGPooling(hidden_units, min_score=0.001, GNN=GCNConv)
        self.conv2 = GINConv(Seq(Lin(hidden_units, hidden_units), ReLU(), Lin(hidden_units, hidden_units)))
        #self.pool2 = SAGPooling(hidden_units, min_score=0.001, GNN=GCNConv)
        self.conv3 = GINConv(Seq(Lin(hidden_units, hidden_units), ReLU(), Lin(hidden_units, hidden_units)))

        self.lin = torch.nn.Linear(hidden_units, 1)

        self.num_features = num_features
        self.num_nodes = num_nodes
        self.min_edges = min_edges


    #Given a batch of states, convert to block-diagonal adjacency, and output Q-value per state
    def forward(self, states):

        num_batches = states.shape[0]

        edge_index, batch = utils.states_to_edge_indicies(states, self.num_nodes, num_batches)
        features = torch.ones((batch.shape[0], self.num_features))


        x = F.relu(self.conv1(features, edge_index))
        #x, edge_index, _, batch, perm, score = self.pool1(
        #    x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        #x, edge_index, _, batch, perm, score = self.pool1(
        #    x, edge_index, None, batch)
        x = F.relu(self.conv3(x, edge_index))

        x = global_max_pool(x, batch)

        x = self.lin(x)

        return x


    #Given the current state, find which new state we should transition to.
    #These new states are the actions we feed to our Q-function
    def act(self, state, epsilon, degree_sequence_dict):

        #Get list of valid states we can transition to from the given state
        valid_actions = utils.getValidActions(state, self.num_nodes, degree_sequence_dict)

        #We are allowed to transition to the same state if we have the minimum number of edges
        if(sum(state) >= self.min_edges):
            valid_actions.append(state)


        #Compute the Q-Value of each valid state we can transition to

        q_values = self.forward(torch.Tensor(valid_actions))


        #Select valid action with max q_value, if greedy-epsilon exploration decides to
        if random.random() > epsilon:

           #print(q_values)
           
           value, valid_action_index = torch.max(q_values, 0)

           action = valid_actions[valid_action_index.item()]


        #With epsilon probability select a random valid action
        else:

            rand_action_index = random.randrange(len(valid_actions))

            action = valid_actions[rand_action_index]



        return action
"""



