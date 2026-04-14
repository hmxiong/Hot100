import torch

def grpo_loss(
    logp_new: torch.Tensor,   # [B, G] 当前策略对每个回答的序列logprob(通常是token logprob求和)
    logp_old: torch.Tensor,   # [B, G] 采样时旧策略logprob
    rewards: torch.Tensor,    # [B, G] 每个prompt下G个回答的reward
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
    logp_ref: torch.Tensor | None = None,  # [B, G] 参考模型logprob(可选)
    eps: float = 1e-8,
):
    # 1) 组内标准化优势 (Group Relative Advantage)
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    adv = adv / (rewards.std(dim=1, keepdim=True) + eps)

    # 2) PPO-style clipped objective
    ratio = torch.exp(logp_new - logp_old)  # [B, G]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3) 可选 KL 正则（约束别偏离参考模型太远）
    if kl_beta > 0 and logp_ref is not None:
        kl = (logp_new - logp_ref).mean()   # 采样近似下常用写法
        loss = policy_loss + kl_beta * kl
    else:
        kl = torch.tensor(0.0, device=policy_loss.device)
        loss = policy_loss

    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach(),
        "adv_mean": adv.mean().detach(),
        "adv_std": adv.std().detach(),
    }