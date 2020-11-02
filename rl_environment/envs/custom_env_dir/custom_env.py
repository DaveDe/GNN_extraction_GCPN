import gym
import torch
import torch.nn.functional as F
import numpy as np

from torch_geometric.utils import dense_to_sparse


class CustomEnv(gym.Env):

    def __init__(self, num_features, blackbox_model, c, max_nodes, min_nodes):

        #Initialize adjacency with all possible nodes
        #Self loops for scaffold nodes
        self.adj = torch.zeros((max_nodes, max_nodes))
        self.adj[0,0] = 1 #Initial node
        self.adj[1,1] = 1 #Scaffold node

        self.num_current_nodes = 2 #Number of nodes in play (including scaffold nodes)

        self.blackbox_model = blackbox_model
        self.c = c #Class we want to learn graph representing
        self.num_features = num_features
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes


    #Reward is computed as average number of connected nodes predicted as class c.
    #Additional reward if the prediction is confident
    """
    def reward(self):

        edge_indices, _ = dense_to_sparse(self.adj)
        features = torch.ones((self.num_current_nodes, self.num_features))

        logits = self.blackbox_model(features, edge_indices)
        probs = F.softmax(logits, dim=1)


        #nodes_with_edges_probs = probs[nodes_with_edges].detach()

        values, indicies = torch.max(probs, 1)

        nodes_predicted_c = (indicies == self.c)

        num_predicted_c = nodes_predicted_c.sum().item()

        reward = (num_predicted_c*10)/(self.num_current_nodes-1)

        return reward
    """

    
    def reward(self):

        #Get black-box labels for all nodes
        edge_indices, _ = dense_to_sparse(self.adj)
        features = torch.ones((self.num_current_nodes, self.num_features))

        logits = self.blackbox_model(features, edge_indices)
        probs = F.softmax(logits, dim=1)

        #Reward is probability of node 0 being predicted as class c
        #reward = probs[0, self.c].detach().item()
        reward = probs[:, self.c].detach().sum().item()#*10#/self.num_current_nodes

        return reward


    def step(self, action):

        #print("Action:", action)

        isDone = False
        isValid = False

        reward = 0

        #coordinates of edge to add
        i = action[0]
        j = action[1]

        #Valid action. This edge doesn't already exist
        if(i != j and self.adj[i,j] != 1):

            #Small reward for valid action
            reward += 0.1

            isValid = True

            #Add edge to graph
            self.adj[i,j] = 1
            self.adj[j,i] = 1

            #If we added an edge to the scaffold node, then add a new disconnected scaffold node
            if(self.adj[self.num_current_nodes-1].sum().item() > 1):

                #Add self-loop to next scaffold node
                if(self.num_current_nodes < self.max_nodes):
                    self.adj[self.num_current_nodes, self.num_current_nodes] = 1
                    self.num_current_nodes += 1
                #Connected to last available node. Exit.
                else:
                    isDone = True

            #Remove self-loops from possibly previously isolated nodes
            self.adj[i,i] = 0
            self.adj[j,j] = 0

        else:

            #small penalty for invalid action
            reward -= 0.1

        if(action[2] == 1 and self.num_current_nodes > self.min_nodes):
            isDone = True

        if(isDone):
            reward += self.reward()


        return self.adj, reward, isDone, isValid



    def reset(self):

        self.adj = torch.zeros((self.max_nodes, self.max_nodes))
        self.adj[0,0] = 1 #Initial node
        self.adj[1,1] = 1 #Scaffold node

        self.num_current_nodes = 2 #Number of nodes in play (including scaffold nodes)


        return self.adj

