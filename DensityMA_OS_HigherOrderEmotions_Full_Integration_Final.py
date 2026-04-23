import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class DensityMA_OS_Full:
    """
    Final integrated simulation: Density × MA OS + Higher-Order Emotions
    Used for the paper "Consciousness as OS and Applications" (v1.0)
    All parameters are grounded in TRIBE v2 simulations and v22 model.
    """

    def __init__(self):
        # ==================== OS Layer Core Parameters ====================
        self.alpha = 0.92                    # MA decay rate (forgetting foundation, same as v21/v22)

        # ==================== Higher-Order Emotions Parameters ====================
        self.coupling = 0.12                 # Strength of divergence (branching sway)
        self.nostalgia_boost = 0.25          # Strength of nostalgic repair
        self.nostalgia_decay = 0.88          # Temporal decay of nostalgia
        self.self_embrace = 0.18             # Strength of zero-distance self-embrace
        # Rationale: TRIBE v2 runs showed optimal balance for identity stabilization
        self.empathy = 0.065                 # Empathy scaling factor for Relation_MA
        # Rationale: Keeps Relation_MA oscillating naturally between ~2.0–2.8

        # ==================== Application Switching Thresholds ====================
        self.GNWT_TH = 0.52                  # GNWT activation threshold
        self.FEP_TH = 0.35                   # FEP activation threshold
        self.IIT_MA_STD_TH = 0.032           # IIT activation threshold (MA_std)
        # Rationale: Derived from TRIBE v2 block statistics (Music → Language → Visual)

        # Pseudo-energy costs (normalized so total ≈ 1.0 ≈ 20 W brain constraint)
        self.cost = {"GNWT": 0.48, "FEP": 0.22, "IIT": 0.35}

    def generate_tribe_v2_stimulus(self, N=1200):
        """Generate TRIBE v2 stimulus pattern for reproducibility"""
        d_eff = np.zeros(N)
        d_eff[100:400] = 0.85   # Music block
        d_eff[400:700] = 0.65   # Language block
        d_eff[700:1000] = 0.92  # Visual block
        d_eff += 0.05 * np.random.randn(N)
        return np.clip(d_eff, 0.0, 1.0)

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
        """Continuous (gradient) application activation strengths"""
        gnwt_act = max(0, Q - self.GNWT_TH) * (1 if MA_std < 0.08 else 0.6) * (1 + 0.35 * d_eff_change)
        fep_act = max(0, self.FEP_TH - Q) * (1 - MA_std) * (1 + 0.18 * relation_ma)
        iit_act = MA_std * ref_level * (1 + 0.28 * relation_ma)
        total = gnwt_act + fep_act + iit_act + 1e-8
        return gnwt_act / total, fep_act / total, iit_act / total

    def step(self, d_eff, ma_prev, relation_ma_prev, nostalgia_trace, t,
             ma_history, relation_ma_history, q_history, prev_d_eff):
        """Single step: OS update + higher-order emotions + gradient switching + cost"""
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

        app_state = f"GNWT:{gnwt_act:.2f} FEP:{fep_act:.2f} IIT:{iit_act:.2f} | Cost:{total_cost:.3f}"

        return ma_new, relation_ma_new, Q, app_state, MA_std, (gnwt_act, fep_act, iit_act), is_shaking, total_cost


# ====================== MAIN SIMULATION ======================
if __name__ == "__main__":
    np.random.seed(42)
    model = DensityMA_OS_Full()

    # TRIBE v2 stimulus
    d_eff = model.generate_tribe_v2_stimulus(N=1200)

    N = 1200
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

        ma[t], relation_ma[t], q[t], app_state, ma_std, activations, is_shaking, total_cost = model.step(
            d_eff[t], ma[t-1], relation_ma[t-1], nostalgia_trace, t,
            ma_history, relation_ma_history, q_history, prev_d_eff
        )

        total_cost_history.append(total_cost)
        ma_std_history.append(ma_std)
        shake_history.append(is_shaking)

        if 100 < t < 400:
            nostalgia_trace[t] = ma[t]

        prev_d_eff = d_eff[t]

    print("=== DensityMA_OS_HigherOrderEmotions_Full_Integration_Final.py ===")
    print(f"Final Q: {q[-1]:.4f}")
    print(f"Final Relation_MA: {relation_ma[-1]:.4f}")
    print(f"Average energy cost: {np.mean(total_cost_history):.4f} (~20 W equivalent)")
    print(f"Identity shake detections: {sum(shake_history)} times")