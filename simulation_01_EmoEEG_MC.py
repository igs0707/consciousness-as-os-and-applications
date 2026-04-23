import numpy as np
import matplotlib.pyplot as plt

class DensityMA_OS_Full:
    """
    DensityMA_OS + Higher-Order Emotions Full Integration
    Simulation 01: EmoEEG-MC (ds005540)
    Emotional context switching (Video/Imagery)
    """
    def __init__(self):
        # OS Layer
        self.alpha = 0.92                    # MA decay with astrocytic consolidation

        # Higher-Order Emotions Parameters
        self.coupling = 0.12
        self.nostalgia_boost = 0.25
        self.nostalgia_decay = 0.88
        self.self_embrace = 0.18
        self.empathy = 0.065

        # Application switching thresholds
        self.GNWT_TH = 0.52
        self.FEP_TH = 0.35
        self.IIT_MA_STD_TH = 0.032

        # Pseudo-energy costs (normalized to ~20 W brain constraint)
        self.cost = {"GNWT": 0.48, "FEP": 0.22, "IIT": 0.35}

    def generate_emo_eeg_mc_stimulus(self, N=2000):
        """Generate stimulus pattern for EmoEEG-MC (ds005540): Emotional context switching between Video and Imagery"""
        np.random.seed(42)
        d_eff = np.zeros(N)
        
        # Emotional blocks with sudden context switches
        blocks = [
            (0, 350, 0.85),    # High positive arousal (Video)
            (350, 650, 0.38),  # Negative context switch
            (650, 950, 0.92),  # Strong positive imagery switch
            (950, 1250, 0.25), # Neutral low phase
            (1250, 1550, 0.78),# Sudden negative emotional surge
            (1550, 2000, 0.62) # Recovery and self-embrace phase
        ]
        
        for start, end, base in blocks:
            length = end - start
            noise = 0.18 * np.random.randn(length)
            d_eff[start:end] = base + noise
        
        # Add sharp transitions to induce Identity Shake
        switch_points = [350, 650, 950, 1250, 1550]
        for p in switch_points:
            if p + 40 < N:
                d_eff[p-30:p+40] += 0.28 * np.random.randn(70)
        
        return np.clip(d_eff, 0.05, 0.98)

    def detect_identity_shake(self, ma_std_history, relation_ma_history, q_history):
        """Detect identity shake (fluctuation of self)"""
        if len(ma_std_history) < 30:
            return False, 0.0
        recent_ma_std = np.mean(ma_std_history[-15:])
        prev_ma_std = np.mean(ma_std_history[-40:-15])
        recent_relation_change = abs(relation_ma_history[-1] - relation_ma_history[-15])
        recent_q_drop = max(0, q_history[-15] - q_history[-1])
        shake_score = (recent_ma_std - prev_ma_std) * 2.5 + recent_relation_change * 1.8 + recent_q_drop * 1.2
        is_shaking = shake_score > 0.45
        return is_shaking, shake_score

    def get_ref_level(self, history_len, is_shaking=False):
        """Reference level (deep reference forced during shake)"""
        if is_shaking:
            return 1.0
        if history_len < 60:
            return 0.3
        elif history_len < 160:
            return 0.65
        return 1.0

    def compute_app_activations(self, Q, MA_std, relation_ma, ref_level, d_eff_change):
        """Continuous gradient application activation strengths"""
        gnwt_act = max(0, Q - self.GNWT_TH) * (1 if MA_std < 0.08 else 0.6) * (1 + 0.35 * d_eff_change)
        fep_act = max(0, self.FEP_TH - Q) * (1 - MA_std) * (1 + 0.18 * relation_ma)
        iit_act = MA_std * ref_level * (1 + 0.28 * relation_ma)
        total = gnwt_act + fep_act + iit_act + 1e-8
        return gnwt_act / total, fep_act / total, iit_act / total

    def step(self, d_eff, ma_prev, relation_ma_prev, nostalgia_trace, t,
             ma_history, relation_ma_history, q_history, prev_d_eff):
        """Single simulation step: OS update + Higher-Order Emotions + gradient switching"""
        d_eff_change = abs(d_eff - prev_d_eff)
        is_shaking, _ = self.detect_identity_shake(ma_history, relation_ma_history, q_history)

        # OS core update
        self_input = self.alpha * ma_prev + (1 - self.alpha) * d_eff
        other_input = self.coupling * 0.85

        # Nostalgic repair
        nostalgia = 0.0
        boost = 1.8 if is_shaking else 1.0
        if t > 650 and len(nostalgia_trace) > 300:
            nostalgia = self.nostalgia_boost * np.mean(nostalgia_trace[100:400]) * \
                        (self.nostalgia_decay ** (t - 650)) * boost

        # Zero-distance self-embrace
        embrace_boost = 1.6 if is_shaking else 1.0
        self_embrace_term = self.self_embrace * ma_prev * embrace_boost

        ma_new = np.clip(self_input + other_input + nostalgia + self_embrace_term, 0.0, 2.0)

        # Relation_MA update
        relation_ma_new = np.clip(relation_ma_prev * 0.94 + self.empathy * ma_new, 0.0, 3.0)

        ref_level = self.get_ref_level(len(ma_history), is_shaking)

        Q = d_eff * ma_new
        MA_std = np.std(ma_history[-80:]) if len(ma_history) > 80 else 0.08

        # Gradient application switching
        gnwt_act, fep_act, iit_act = self.compute_app_activations(
            Q, MA_std, relation_ma_new, ref_level, d_eff_change
        )

        # Normalized total energy cost (~20 W)
        total_cost = (gnwt_act * self.cost["GNWT"] +
                      fep_act * self.cost["FEP"] +
                      iit_act * self.cost["IIT"])

        return ma_new, relation_ma_new, Q, MA_std, (gnwt_act, fep_act, iit_act), is_shaking, total_cost


# ====================== MAIN SIMULATION ======================
if __name__ == "__main__":
    print("=== Simulation 01: EmoEEG-MC (ds005540) ===")
    print("Emotional context switching (Video/Imagery) - DensityMA_OS + Higher-Order Emotions")
    print("Testing robustness of zero-distance self-embrace loop under rapid emotional shifts\n")

    model = DensityMA_OS_Full()
    N = 2000
    d_eff = model.generate_emo_eeg_mc_stimulus(N)

    # Initialize arrays
    ma = np.zeros(N)
    q = np.zeros(N)
    relation_ma = np.zeros(N)
    total_cost_history = []
    ma_std_history = []
    ma_history = []
    relation_ma_history = []
    q_history = []
    nostalgia_trace = np.zeros(N)
    shake_history = []

    ma[0] = d_eff[0]
    relation_ma[0] = 0.12
    prev_d_eff = d_eff[0]

    for t in range(1, N):
        ma_history.append(ma[t-1])
        relation_ma_history.append(relation_ma[t-1])
        q_history.append(q[t-1])

        ma[t], relation_ma[t], q[t], ma_std, activations, is_shaking, total_cost = model.step(
            d_eff[t], ma[t-1], relation_ma[t-1], nostalgia_trace, t,
            ma_history, relation_ma_history, q_history, prev_d_eff
        )

        total_cost_history.append(total_cost)
        ma_std_history.append(ma_std)
        shake_history.append(is_shaking)

        # Record nostalgia trace during emotional peaks
        if 100 < t < 600 or 1200 < t < 1700:
            nostalgia_trace[t] = ma[t]

        prev_d_eff = d_eff[t]

    # Final results
    print("Simulation completed successfully!")
    print(f"Final Q: {q[-1]:.4f}")
    print(f"Final Relation_MA: {relation_ma[-1]:.4f}")
    print(f"Average Energy Cost: {np.mean(total_cost_history):.4f} (~20 W equivalent)")
    print(f"Identity Shake detections: {sum(shake_history)} times")
    print(f"Max MA_std: {max(ma_std_history):.4f}")

    # Save high-resolution graph for paper
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('Simulation 01: EmoEEG-MC (ds005540)\n'
                 'DensityMA_OS + Higher-Order Emotions (Self-Embrace Loop)', fontsize=14, fontweight='bold')

    axs[0].plot(d_eff, color='blue', label='D_eff (Emotional context switching)')
    axs[0].set_ylabel('Input Density D_eff')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(q, color='purple', label='Raw Qualia Q(t)')
    axs[1].set_ylabel('Raw Qualia Q')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(relation_ma, color='orange', label='Relation_MA (Self-Embrace)')
    axs[2].set_xlabel('Time steps')
    axs[2].set_ylabel('Relation_MA')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_EmoEEG_MC_Simulation_Result.png', dpi=300, bbox_inches='tight')
    print("\n✅ Graph saved as '01_EmoEEG_MC_Simulation_Result.png'")
    print("Ready for next simulation (2/17).")