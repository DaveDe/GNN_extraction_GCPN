- Is it ok to add self-loop to isolated nodes??

- Is it ok that we are taking the mean of each action components distribution for log prob and entropy calculations?

- Invalid actions??

	- In this setup, first node selection can't possibly be invalid
	- 2nd node selection invalid if we select the same node as first, or an existing edge
	- selecting to end before min number of edges is invalid


To
demonstrate the benefits of learning-based approaches, we further implement a simple rule based
model using the stochastic hill-climbing algorithm. We start with a graph containing a single atom
(the same setting as GCPN), traverse all valid actions given the current state, randomly pick the next
state with top 5 highest property score as long as there is improvement over the current state, and loop
until reaching the maximum number of nodes