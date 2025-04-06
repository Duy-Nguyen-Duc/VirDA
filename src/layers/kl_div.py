import torch.nn.functional as F

def kl_divergence_loss(student_logits, teacher_logits):
    """
    Compute KL divergence loss between student and teacher outputs.

    Args:
        student_logits (Tensor): Raw logits from the student model.
        teacher_logits (Tensor): Raw logits from the teacher model.
    
    Returns:
        Tensor: The KL divergence loss (averaged over the batch).
    """
    # Convert student logits to log probabilities.
    student_log_probs = F.log_softmax(student_logits, dim=1)
    # Convert teacher logits to probabilities.
    teacher_probs = F.softmax(teacher_logits, dim=1)
    # Compute the KL divergence loss.
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return loss
