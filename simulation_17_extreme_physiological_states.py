import numpy as np
import matplotlib.pyplot as plt

class DensityMA_OS_Full:
    """
    DensityMA_OS + Higher-Order Emotions Full Integration
    Simulation 17: Extreme Physiological States
    Combined: Birth, Sexual Intimacy, and Acute Illness
    """
    def __init__(self):
        # OS Layer
        self.alpha = 0.92
        # Higher-Order Emotions Parameters
        self.coupling = 0.12
        self.nostalgia_boost = 0.25
        self.nostalgia_decay = 0.88
        self.self_embrace = 0.18
        self.empathy = 0.065
        # Application thresholds
        self.GNWT_TH = 0.52
        self.FEP_TH = 0.35
        self.IIT_MA_STD_TH = 0.032
        # Energy costs (~20 W)
        self.cost = {"GNWT": 0.48, "FEP": 0.22, "IIT": 0.35}

    def generate_birth_stimulus(self, N=2000):
        """Birth: Intense pain + oxytocin surge + mother-child bonding"""
        np.random.seed(42)
        d_eff = np.zeros(N)
        d_eff[0:600] = 0.65 + 0.25 * np.random.randn(600)   # labor build-up
        d_eff[600:1100] = 0.95 + 0.18 * np.random.randn(500) # delivery peak
        d_eff[1100:2000] = 0.72 + 0.15 * np.random.randn(900) # bonding & recovery
        return np.clip(d_eff, 0.25, 0.98)

    def generate_sexual_intimacy_stimulus(self, N=2000):
        """Sexual intimacy: High arousal + strong joint agency"""
        np.random.seed(42)
        d_eff = np.zeros(N)
        d_eff[0:400] = 0.55 + 0.20 * np.random.randn(400)
        d_eff[400:900] = 0.92 + 0.22 * np.random.randn(500)  # peak arousal
        d_eff[900:2000] = 0.68 + 0.18 * np.random.randn(1100) # resolution
        return np.clip(d_eff, 0.30, 0.98)

    def generate_acute_illness_stimulus(self, N=2000):
        """Acute illness: Fever, pain, malaise"""
        np.random.seed(42)
        d_eff = np.zeros(N)
        d_eff[:] = 0.38 + 0.12 * np.random.randn(N)
        illness_peaks = [350, 720, 1100, 1480]
        for start in illness_peaks:
            if start + 120 < N:
                d_eff[start:start+120] += 0.35 * np.exp(-0.025 * np.arange(120))
        return np.clip(d_eff, 0.15, 0.72)

    # (detect_identity_shake, get_ref_level, compute_app_activations, step は省略せず完全コピー)
    # 前のコードと同じ内容をここに貼り付け（スペースの都合上省略表記していますが、実際のファイルでは完全に入れてください）

    def detect_identity_shake(self, ma_std_history, relation_ma_history, q_history):
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
        if is_shaking:
            return 1.0
        if history_len < 60:
            return 0.3
        elif history_len < 160:
            return 0.65
        return 1.0

    def compute_app_activations(self, Q, MA_std, relation_ma, ref_level, d_eff_change):
        gnwt_act = max(0, Q - self.GNWT_TH) * (1 if MA_std < 0.08 else 0.6) * (1 + 0.35 * d_eff_change)
        fep_act = max(0, self.FEP_TH - Q) * (1 - MA_std) * (1 + 0.18 * relation_ma)
        iit_act = MA_std * ref_level * (1 + 0.28 * relation_ma)
        total = gnwt_act + fep_act + iit_act + 1e-8
        return gnwt_act / total, fep_act / total, iit_act / total

    def step(self, d_eff, ma_prev, relation_ma_prev, nostalgia_trace, t,
             ma_history, relation_ma_history, q_history, prev_d_eff):
        d_eff_change = abs(d_eff - prev_d_eff)
        is_shaking, _ = self.detect_identity_shake(ma_history, relation_ma_history, q_history)

        self_input = self.alpha * ma_prev + (1 - self.alpha) * d_eff
        other_input = self.coupling * 0.85

        nostalgia = 0.0
        boost = 1.8 if is_shaking else 1.0
        if t > 650 and len(nostalgia_trace) > 300:
            nostalgia = self.nostalgia_boost * np.mean(nostalgia_trace[100:400]) * \
                        (self.nostalgia_decay ** (t - 650)) * boost

        embrace_boost = 1.6 if is_shaking else 1.0
        self_embrace_term = self.self_embrace * ma_prev * embrace_boost

        ma_new = np.clip(self_input + other_input + nostalgia + self_embrace_term, 0.0, 2.0)
        relation_ma_new = np.clip(relation_ma_prev * 0.94 + self.empathy * ma_new, 0.0, 3.0)

        ref_level = self.get_ref_level(len(ma_history), is_shaking)
        Q = d_eff * ma_new
        MA_std = np.std(ma_history[-80:]) if len(ma_history) > 80 else 0.08

        gnwt_act, fep_act, iit_act = self.compute_app_activations(
            Q, MA_std, relation_ma_new, ref_level, d_eff_change)

        total_cost = (gnwt_act * self.cost["GNWT"] + fep_act * self.cost["FEP"] + iit_act * self.cost["IIT"])

        return ma_new, relation_ma_new, Q, MA_std, (gnwt_act, fep_act, iit_act), is_shaking, total_cost


# ====================== MAIN SIMULATION ======================
if __name__ == "__main__":
    print("=== Simulation 17: Extreme Physiological States ===")
    print("Birth / Sexual Intimacy / Acute Illness")
    print("Final test of self-embrace in life-critical physiological extremes\n")

    model = DensityMA_OS_Full()
    N = 2000

    conditions = {
        "Birth": model.generate_birth_stimulus,
        "Sexual_Intimacy": model.generate_sexual_intimacy_stimulus,
        "Acute_Illness": model.generate_acute_illness_stimulus
    }

    for name, generator in conditions.items():
        print(f"\n--- Running {name} ---")
        d_eff = generator(N)

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

            if t > 400:
                nostalgia_trace[t] = ma[t]

            prev_d_eff = d_eff[t]

        print(f"  Final Relation_MA : {relation_ma[-1]:.4f}")
        print(f"  Avg Energy Cost   : {np.mean(total_cost_history):.4f}")
        print(f"  Shake detections  : {sum(shake_history)} times")

        # Graph
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f'Simulation 17: {name}\nDensityMA_OS + Higher-Order Emotions', fontsize=14, fontweight='bold')
        axs[0].plot(d_eff, color='blue', label=f'D_eff ({name})')
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
        plt.savefig(f'17_{name}_Simulation_Result.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Graph saved as '17_{name}_Simulation_Result.png'")

    print("\n🎉 All 17 simulations completed!")
    print("これで全17データセットのコードが揃いました！")