"""
Knowledge Distillation Engine
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 07)
Formula: Loss = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(Softmax(student/T) || Softmax(teacher/T))
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard ground-truth Cross-Entropy
    and soft dark-knowledge KL-divergence matching at temperature T.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard Cross-Entropy on ground truth
        hard_loss = self.ce(student_logits, targets)

        # Soft predictions scaled by temperature T
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)

        # KL divergence scaled by T^2 to preserve gradient magnitude
        soft_loss = self.kl(soft_student, soft_teacher) * (self.T ** 2)

        total_loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
        return total_loss


def train_student_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    temperature: float = 4.0,
    alpha: float = 0.5,
    learning_rate: float = 1e-3,
    lr: float = None,
    device: str = 'cpu'
) -> List[float]:
    """Trains compact student supervised by both ground truth labels and teacher soft targets."""
    if lr is not None:
        learning_rate = lr
    student.train()
    student.to(device)
    teacher.eval()
    teacher.to(device)

    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)

            loss = criterion(student_logits, teacher_logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        epoch_loss = running_loss / max(1, batches)
        loss_history.append(epoch_loss)

    return loss_history


def train_student_without_distillation(
    student: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    lr: float = None,
    device: str = 'cpu'
) -> List[float]:
    """Trains compact student from scratch with plain Cross-Entropy on ground truth."""
    if lr is not None:
        learning_rate = lr
    student.train()
    student.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            student_logits = student(x)
            loss = criterion(student_logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        epoch_loss = running_loss / max(1, batches)
        loss_history.append(epoch_loss)

    return loss_history
