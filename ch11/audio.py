"""Audio representations: waveform, STFT, mel spectrogram, Griffin-Lim inversion, and a tiny VQ neural codec."""
import json, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
SR = 16000
def demo_signal(seconds=1.5, seed=0):
    t = np.arange(int(SR * seconds)) / SR; rng = np.random.default_rng(seed)
    chord = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0)) * (t < 0.5)            # C major chord
    chirp = np.sin(2 * np.pi * (300 * (t - 0.5) + 1500 * (t - 0.5) ** 2)) * ((t >= 0.5) & (t < 1.0))  # rising tone
    burst = rng.normal(0, 0.6, len(t)) * ((t >= 1.1) & (t < 1.25))                                   # noise burst
    x = 0.25 * (chord + chirp + burst); return x.astype(np.float32)
def stft(x, n_fft=512, hop=128): return torch.stft(torch.tensor(x), n_fft, hop, window=torch.hann_window(n_fft), return_complex=True)
def mel_filterbank(n_mels=64, n_fft=512, sr=SR):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700); mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)); bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        fb[m - 1, l:c] = (np.arange(l, c) - l) / max(c - l, 1); fb[m - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return torch.tensor(fb, dtype=torch.float32)
def griffin_lim(mag, n_iter=60, n_fft=512, hop=128, length=None):
    """Recover a waveform from a magnitude spectrogram by iteratively estimating phase."""
    g = torch.Generator().manual_seed(0); angle = torch.exp(2j * math.pi * torch.rand(mag.shape, generator=g))
    for _ in range(n_iter):
        x = torch.istft(mag * angle, n_fft, hop, window=torch.hann_window(n_fft), length=length)
        angle = torch.exp(1j * torch.angle(torch.stft(x, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True)))
    return torch.istft(mag * angle, n_fft, hop, window=torch.hann_window(n_fft), length=length)
def spectral_convergence(x, y, n_fft=512, hop=128):
    X, Y = stft(x, n_fft, hop).abs(), stft(y, n_fft, hop).abs(); return float(torch.norm(X - Y) / torch.norm(X))
def snr_db(ref, est): return float(10 * np.log10((ref ** 2).sum() / ((ref - est) ** 2).sum()))

class VQCodec(nn.Module):
    """Encoder -> nearest codebook vector per frame (a discrete token) -> decoder. 1 token per 8 ms."""
    def __init__(s, frame=128, d=32, K=256):
        super().__init__(); s.frame = frame
        s.enc = nn.Sequential(nn.Conv1d(1, 64, 8, 4, 2), nn.GELU(), nn.Conv1d(64, 64, 8, 4, 2), nn.GELU(), nn.Conv1d(64, d, 8, 8, 0))
        s.dec = nn.Sequential(nn.ConvTranspose1d(d, 64, 8, 8), nn.GELU(), nn.ConvTranspose1d(64, 64, 8, 4, 2), nn.GELU(), nn.ConvTranspose1d(64, 1, 8, 4, 2))
        s.codebook = nn.Embedding(K, d)
    def forward(s, x):
        z = s.enc(x).transpose(1, 2)                                                  # (B, frames, d)
        dist = (z.unsqueeze(2) - s.codebook.weight).pow(2).sum(-1); codes = dist.argmin(-1)
        q = s.codebook(codes); q_st = z + (q - z).detach()                            # straight-through estimator
        out = s.dec(q_st.transpose(1, 2))
        return out, codes, F.mse_loss(q, z.detach()) + 0.25 * F.mse_loss(z, q.detach())
def training_clip(rng, n=4096):
    t = np.arange(n) / SR; x = np.zeros(n)
    for _ in range(rng.integers(1, 4)):
        f0 = rng.uniform(100, 2000); x += rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * (f0 * t + rng.uniform(-300, 300) * t * t) + rng.uniform(0, 6))
    x *= np.minimum(1, np.linspace(0, 4, n)) * np.minimum(1, np.linspace(4, 0, n)); x += rng.normal(0, 0.01, n)
    return x.astype(np.float32)

if __name__ == "__main__":
    res = {}; x = demo_signal(); np.save("demo.npy", x)
    S = stft(x); mag = S.abs(); fb = mel_filterbank(); mel = fb @ mag.pow(2)
    res["shapes"] = {"samples": len(x), "stft": list(S.shape), "mel": list(mel.shape)}
    torch.save({"mag": mag, "mel": mel}, "spec.pt")
    y = griffin_lim(mag, length=len(x)).numpy()
    res["griffin_lim"] = {"spectral_convergence": spectral_convergence(x, y), "snr_db": snr_db(x, y)}
    mu = 255; xq = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu); xr = np.sign(xq) * ((1 + mu) ** np.abs(np.round(xq * 127) / 127) - 1) / mu
    res["mu_law_8bit"] = {"kbps": SR * 8 / 1000, "snr_db": snr_db(x, xr)}
    print(res)
    torch.manual_seed(0); rng = np.random.default_rng(0); codec = VQCodec(); opt = torch.optim.Adam(codec.parameters(), 1e-3); t0 = time.time()
    for step in range(1500):
        xb = torch.tensor(np.stack([training_clip(rng) for _ in range(16)]))[:, None]
        out, codes, vq = codec(xb); loss = F.mse_loss(out, xb) + vq
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:                                                           # revive unused codes
            used = codes.unique(); dead = torch.tensor([k for k in range(256) if k not in used.tolist()])
            if len(dead): codec.codebook.weight.data[dead] = codec.enc(xb).transpose(1, 2).reshape(-1, 32)[torch.randint(0, codes.numel(), (len(dead),))].detach()
    with torch.no_grad():
        test = torch.tensor(np.stack([training_clip(np.random.default_rng(100 + i)) for i in range(32)]))[:, None]
        out, codes, _ = codec(test)
    frames_per_s = SR / 128
    res["vq_codec"] = {"tokens_per_second": frames_per_s, "kbps": frames_per_s * 8 / 1000, "snr_db": snr_db(test.numpy(), out.numpy()),
                       "codes_used": int(codes.unique().numel()), "seconds": time.time() - t0, "compression_vs_16bit_pcm": SR * 16 / (frames_per_s * 8)}
    torch.save({"orig": test[0, 0, :1024], "rec": out[0, 0, :1024]}, "codec_example.pt")
    print(res["vq_codec"]); json.dump(res, open("audio.json", "w"), indent=1)
