import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
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


class DDINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.register_buffer('center_grid', torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_epochs),
            np.ones(nepochs - warmup_teacher_epochs) * teacher_temp
        ))

        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, student_outputs: Tensor, teacher_outputs: Tensor, epoch: int):
        student_cls_pred, student_region_pred, student_feats, student_npatch = student_outputs
        teacher_cls_pred, teacher_region_pred, teacher_feats, teacher_npatch = teacher_outputs

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_cls = self.softmax((teacher_cls_pred - self.center) / temp)
        teacher_cls = teacher_cls.detach().chunk(2)

        teacher_region = self.softmax((teacher_region_pred - self.center_grid) / temp)
        teacher_region = teacher_region.detach().chunk(2)
        
        teacher_feats = teacher_feats.chunk(2)

        N = teacher_npatch[0]   # number of patches in the first view
        B = teacher_region[0].shape[0] // N

        # student sharpening
        student_cls = student_cls_pred / self.student_temp
        student_cls = student_cls.chunk(self.ncrops)

        student_region = student_region_pred / self.student_temp
        student_split_size = [student_npatch[0]] * 2 + [student_npatch[1]] * (self.ncrops - 2)
        student_split_size_bs = [i * B for i in student_split_size]
        student_region = torch.split(student_region, student_split_size_bs, dim=0)
        
        student_feats = torch.split(student_feats, student_split_size_bs, dim=0)

        total_loss, n_loss_terms = 0, 0

        for i, q in enumerate(teacher_cls):
            for j, v in enumerate(student_cls):
                if j == i:
                    continue

                # view level prediction loss
                loss = 0.5 * torch.sum(-q * self.logsoftmax(v), dim=-1)

                # region level prediction loss
                s_region_cur = student_region[j].view(B, student_split_size[j], -1)
                s_fea_cur = student_feats[j].view(B, student_split_size[j], -1)

                t_region_cur = teacher_region[i].view(B, N, -1)
                t_fea_cur = teacher_feats[i].view(B, N, -1)

                # similarity matrix between two sets of region features
                region_sim_matrix = torch.matmul(F.normalize(s_fea_cur, p=2, dim=-1), F.normalize(t_fea_cur, p=2, dim=-1).permute(0, 2, 1))
                region_sim_ind = region_sim_matrix.max(dim=2)[1]

                t_indexed_region = torch.gather(t_region_cur, 1, region_sim_ind.unsqueeze(2).expand(-1, -1, t_region_cur.shape[2]))

                loss += 0.5 * torch.sum(-t_indexed_region * self.logsoftmax(s_region_cur), dim=-1).mean(-1)

                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_cls_pred, teacher_region_pred)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_pred, teacher_grid_pred):
        # view level center update
        batch_center = torch.sum(teacher_pred, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center /= len(teacher_pred) * dist.get_world_size()

        # region level center update
        batch_grid_center = torch.sum(teacher_grid_pred, dim=0, keepdim=True)
        dist.all_reduce(batch_grid_center)
        batch_grid_center /= len(teacher_grid_pred) * dist.get_world_size()

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center_grid = self.center_grid * self.center_momentum + batch_grid_center * (1 - self.center_momentum)
