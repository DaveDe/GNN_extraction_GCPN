- Is it ok to add self-loop to isolated nodes??

- Is it ok that we are taking the mean of each action components distribution for log prob and entropy calculations?

** What do we backprop through? Just critic value?

	- We are not saving gradients from computing log prob of categorical dist. Is this because of categorical?