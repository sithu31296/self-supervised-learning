import torch
import numpy as np
from torch import nn, Tensor
from torch import distributed as dist


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_epochs),
            np.ones(nepochs - warmup_teacher_epochs) * teacher_temp
        ))

        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, student_pred: Tensor, teacher_pred: Tensor, epoch: int):
        student_pred = student_pred / self.student_temp
        student_pred = student_pred.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_pred = self.softmax((teacher_pred - self.center) / temp)
        teacher_pred = teacher_pred.detach().chunk(2)

        total_loss, n_loss_terms = 0, 0

        for i, q in enumerate(teacher_pred):
            for j, v in enumerate(student_pred):
                if j == i:
                    continue

                loss = torch.sum(-q * self.logsoftmax(v), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_pred)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_pred):
        batch_center = torch.sum(teacher_pred, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center /= len(teacher_pred) * dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
