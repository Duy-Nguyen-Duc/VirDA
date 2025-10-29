import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import get_backbone
from .prompt import InstancewiseVisualPromptCoordNet
from .head import Classifier, DomainDiscriminator
from .utils import freeze_layers, grad_reverse

class UModel(nn.Module):
    def __init__(
        self, 
        backbone="vit_b_32",
        hidden_dim=256,
        out_dim=65,
        imgsize=384, 
        layer = 6, 
        patch_size = 32,
        freeze_backbone=True
    ):
        super(UModel, self).__init__()
        self.backbone = get_backbone(backbone)
        self.in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.out_dim = out_dim

        if freeze_backbone:
            freeze_layers([self.backbone])

        self.src_vr = InstancewiseVisualPromptCoordNet(size = imgsize, layers=layer, patch_size=patch_size, channels=3, dropout_p=0.0)
        self.tgt_vr = InstancewiseVisualPromptCoordNet(size = imgsize, layers=layer, patch_size=patch_size, channels=3, dropout_p=0.0)
        self.src_cls = Classifier(in_dim=self.in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=0.3)
        self.tgt_cls = Classifier(in_dim=self.in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=0.3)
        self.domain_discriminator = DomainDiscriminator(in_dim=self.in_dim, hidden_dim=hidden_dim, out_dim=2, dropout=0.5)

    def forward(self, x, vr_branch, head_branch, grl_alpha=None, M=0):
        prompt = self.src_vr if vr_branch == "src" else self.tgt_vr if vr_branch == "tgt" else None
        head = self.src_cls if head_branch == "src" else self.tgt_cls if head_branch == "tgt" else self.domain_discriminator

        x_prompt = prompt(x) if prompt is not None else x
        feats = self.backbone(x_prompt)
        
        if head_branch == "domain" and grl_alpha is not None:
            feats = grad_reverse(feats, grl_alpha)
        
        if M == 0:
            return head(feats)
        else:
            probs = []
            mean_logits = torch.zeros(x.shape[0], self.out_dim, device=x.device)
            for _ in range(M):
                logits = head(feats)
                mean_logits += logits
                probs.append(F.softmax(logits, dim=-1))
            P = torch.stack(probs, dim=0)
            p_mean = P.mean(dim=0)
            H_pred = self._entropy(p_mean, dim=-1)
            H_exp = self._entropy(P, dim=-1).mean(dim=0)
            mi = H_pred - H_exp

            return mean_logits / M, mi
        
    @staticmethod
    def _entropy(p, dim, eps=1e-10):
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=dim)

class EigenCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handle = None
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def register_hook(self):
        if self.hook_handle is None:
            self.hook_handle = self.target_layer.register_forward_hook(self.save_activation)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def __call__(self, **kwargs):
        self.activations = None
        
        with torch.no_grad():
            _ = self.model(**kwargs)
        
        if self.activations is None:
            raise RuntimeError("No activations captured. Check target layer.")
        
        activations = self.activations
        B, N, C = activations.shape
        Hp, Wp = 12, 12
        
        if N > 1:
            activations = activations[:, 1:, :]
            N = activations.shape[1]
        with torch.amp.autocast("cuda",enabled=False):
            X = activations.transpose(1, 2).float()
            Xc = X - X.mean(dim=2, keepdim=True)
            cov = torch.bmm(Xc, Xc.transpose(1, 2)) / (N - 1 + 1e-8)
            eye = torch.eye(C, device=cov.device, dtype=cov.dtype).unsqueeze(0)
            cov = cov + eye * 1e-4

            evals, evecs = torch.linalg.eigh(cov)
            w = evecs[:, :, -1]
            proj_mean = (w.unsqueeze(-1) * Xc).sum(dim=1).mean(dim=-1)
            sgn = torch.sign(proj_mean)
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            w = w * sgn.unsqueeze(-1)
            M = torch.einsum('bc,bcn->bn', w, X)            # [B,N]
            M = F.relu(M)
        M = M.view(B, 1, Hp, Wp)
        M_min = M.amin(dim=(2, 3), keepdim=True)
        M_max = M.amax(dim=(2, 3), keepdim=True)
        M = (M - M_min) / (M_max - M_min + 1e-6)
        return M
