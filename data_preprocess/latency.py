import time
import torch


class DistMetrics:
    def _kl_div(self, p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nan_to_num(q.float(), nan=0.0, posinf=0.0, neginf=0.0)

        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        q = q / (q.sum(dim=-1, keepdim=True) + eps)

        # 避免 log(0)
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)

        # 这里用log2 保证JSD的输出在[0, 1]
        # 如果用e为底， JSD的输出在[0, ln2]
        kl = torch.sum(p * (p.log2() - q.log2()), dim=-1)
        return kl

    # JSD分布 -》 KL散度的平滑版，具有对称性
    def _js_div(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.to(p.device)
        m = 0.5 * (p + q)
        jsd = 0.5 * self._kl_div(p, m) + 0.5 * self._kl_div(q, m)
        return jsd

    # L1范数
    def _l1_norm(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.to(p.device)
        p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nan_to_num(q.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # [0, 2]
        l1_diff = torch.abs(p - q).sum(dim=-1)

        return 0.5 * l1_diff

    # 余弦相似度（返回 1 - cos）
    def _cos_sim(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.to(p.device)

        p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.nan_to_num(q.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # 点积
        dot = torch.sum(p * q, dim=-1)

        # L2 范数
        p_norm = torch.linalg.norm(p, dim=-1)
        q_norm = torch.linalg.norm(q, dim=-1)

        denom = p_norm * q_norm + 1e-8

        # 余弦相似度
        cos_sim = dot / denom

        # 如果你确实需要 1 - cos
        return 1 - cos_sim


def benchmark_once(metric_fn, p, q, device, warmup=5, iters=50, name="metric"):
    # 预热，避免第一次调用过慢影响统计
    with torch.no_grad():
        for _ in range(warmup):
            out = metric_fn(p, q)
        if device == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            out = metric_fn(p, q)
        if device == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()

    total = end - start
    per_call = total / iters
    print(f"{name:8s} total: {total:.6f} s, per call: {per_call * 1000:.4f} ms")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 随机生成 50 对 32000 维“概率分布”
    # 这里直接用正随机数，归一化在 _kl_div 里已经做了
    p = torch.rand(50, 32000, device=device)
    q = torch.rand(50, 32000, device=device)

    metrics = DistMetrics()

    # 分别对四个函数做 benchmark
    benchmark_once(metrics._kl_div,  p, q, device, name="KL")
    benchmark_once(metrics._js_div,  p, q, device, name="JSD")
    benchmark_once(metrics._l1_norm, p, q, device, name="L1")
    benchmark_once(metrics._cos_sim, p, q, device, name="COS(1-)")

if __name__ == "__main__":
    main()