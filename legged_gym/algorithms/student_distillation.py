import torch


class StudentDistillation:
    """ Distillation algorithm: a depth-camera + GRU student policy imitates the frozen teacher's
        mean action via MSE, with truncated BPTT over rollout chunks. Plays the same structural
        role here as rsl_rl.algorithms.PPO does for the teacher (owns the trainable network, its
        optimizer, and the gradient step) but for supervised action regression instead of a PPO
        surrogate loss. See StudentDistillationRunner.learn_distill for the rollout loop that
        drives it - this class only owns the network/optimizer/hidden-state, not the env loop.
    """

    def __init__(self, student, learning_rate=1e-3, max_grad_norm=1.0, device='cpu'):
        self.device = device
        self.student = student.to(device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        self.hidden = None

    def init_hidden(self, num_envs):
        self.hidden = self.student.init_hidden(num_envs, self.device)

    def reset_hidden(self, dones):
        self.hidden = self.student.reset_hidden(self.hidden, dones)

    def act(self, depth_img, proprio, near, far):
        action, self.hidden = self.student.forward_step(depth_img, proprio, self.hidden, near, far)
        return action

    def set_learning_rate(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def update(self, chunk_loss):
        self.optimizer.zero_grad()
        chunk_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.hidden = self.hidden.detach()  # bounds backprop to this chunk (truncated BPTT)
        return grad_norm
