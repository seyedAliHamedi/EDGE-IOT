
import torch
class SharedAdam(torch.optim.Adam):
    """
    Adam optimizer with shared states for multiprocessing environments.
    This allows parameters like step, exponential moving averages, and squared averages
    to be shared across multiple processes.
    """

    def __init__(self, params, lr=0.005):
        """
        Initializes the SharedAdam optimizer.
        
        Args:
        - params: Parameters to optimize.
        - lr: Learning rate for the optimizer.
        """
        # Initialize with standard Adam optimizer
        super(SharedAdam, self).__init__(params, lr=lr)
        
        # Share memory across processes for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    # Initialize shared state
                    state = self.state[p]
                    state['step'] = torch.tensor(0.0).share_memory_()  # Shared step counter
                    state['exp_avg'] = torch.zeros_like(p.data).share_memory_()  # Shared exponential moving average
                    state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()  # Shared squared exponential average

    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
        - closure: A closure that re-evaluates the model and returns the loss (optional).
        """
        # Iterate over parameter groups and apply optimization step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue  # Skip parameters with no gradients

                # Retrieve the shared state for this parameter
                state = self.state[p]
                
                # Increment the shared step count
                state['step'] += 1  
        
        # Call the original Adam step to perform weight updates
        super(SharedAdam, self).step(closure)
