- Is it ok to add self-loop to isolated nodes??


- Invalid actions??

	- In this setup, first node selection can't possibly be invalid
	- 2nd node selection invalid if we select the same node as first, or an existing edge
	- selecting to end before min number of edges is invalid




***** Try black-box ripper for GNNs

		- Train generator on some classic graph dataset....


	Consider this as more of an exploratory/benchmarking paper, simply providing the results of various methods,
	and weighing fidelity vs number queries

	- Can simply try blackbox ripper and compare with DQN. But our DQN approach can't generate features, which is why we prefer GCPN to work. Unless we decompose DQN action space similar to how adversarial attack on graph paper did.

	- Evolutionary algo to search latent space of generator to find high confidence samples, vs RL generator which rewards directly constructing a high confidence sample.

