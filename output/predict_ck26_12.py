"""
CK26-12 마찰계수·마모량 예측 모델
- 학습 데이터: 25년 CK10, CK11, CK12 + 26년 CK10, CK11
- 예측 대상: 26년 CK12 (마찰계수 시계열 + 최종 마모량)
"""

import pandas as pd
import numpy as np

# ──────────────────────────────────────────
# 1. 파일 경로 설정
# ──────────────────────────────────────────
BASE = "/root/.claude/uploads/83d357dc-79f5-5c5c-a583-9edc283e9b63/"
FILES = {
    "25_ck10": BASE + "8a7f86c7-20250331_200N_2.5hz_11mm_30min_ck101.TSV",
    "25_ck11": BASE + "a1a306ed-20250331_200N_2.5hz_11mm_30min_ck111.TSV",
    "25_ck12": BASE + "9ed685a2-20250331_200N_2.5hz_11mm_30min_ck121re.TSV",
    "26_ck10": BASE + "89830de2-20260430_Joguang_CK26_101.TSV",
    "26_ck11": BASE + "ee13945f-20260518_Joguang_CK2611_1.TSV",
}

# PDF에서 확인된 실측 마모량 (μm²)
WEAR_AREA = {
    "25_ck10": 142020,
    "25_ck11": 189040,   # 조기종료 → 마모 더 심함
    "25_ck12": 189040,   # 조기종료 (CK11과 동급으로 추정, 실측 동일 기재)
    "26_ck10": None,     # 아직 미확인
    "26_ck11": None,
}

# ──────────────────────────────────────────
# 2. TSV 파싱
# ──────────────────────────────────────────
def parse_tsv(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if '\tTime\t' in l)
    header = lines[header_idx].strip().split('\t')
    rows = []
    for l in lines[header_idx+1:]:
        s = l.strip()
        if not s:
            continue
        if s.startswith(('Prog', 'Test', 'High')):
            continue
        parts = s.split('\t')
        if len(parts) >= 12:
            rows.append(parts)
    df = pd.DataFrame(rows)
    n = min(len(header), len(df.columns))
    df = df.iloc[:, :n]
    df.columns = header[:n]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Friction Coefficient', 'Test Time'])
    return df.reset_index(drop=True)

# ──────────────────────────────────────────
# 3. 피처 추출
# ──────────────────────────────────────────
def extract_features(df, name=""):
    fc = df['Friction Coefficient'].values
    t  = df['Test Time'].values

    shutdown = len(df) < 1000  # 조기종료 여부

    # 전체 통계
    fc_mean  = np.mean(fc)
    fc_std   = np.std(fc)
    fc_max   = np.max(fc)
    fc_min   = np.min(fc)

    # 초기 구간 (처음 30초 or 전체 50% 중 작은 것)
    early_mask = t <= min(t[-1] * 0.15, 60)
    if early_mask.sum() == 0:
        early_mask = np.ones(len(t), dtype=bool)
    fc_early_mean = np.mean(fc[early_mask])
    fc_early_max  = np.max(fc[early_mask])

    # 초기 상승률 (ramp rate): 처음 5~30초 구간 기울기
    ramp_mask = (t >= 5) & (t <= 30)
    if ramp_mask.sum() >= 2:
        ramp_slope = np.polyfit(t[ramp_mask], fc[ramp_mask], 1)[0]
    else:
        ramp_slope = np.polyfit(t[:max(2, len(t)//4)],
                                fc[:max(2, len(t)//4)], 1)[0]

    # 중반 / 후반 평균 (완전 시험만)
    if not shutdown:
        mid_mask  = (t >= t[-1]*0.3) & (t <= t[-1]*0.6)
        late_mask = t >= t[-1]*0.7
        fc_mid_mean  = np.mean(fc[mid_mask])  if mid_mask.sum()  else fc_mean
        fc_late_mean = np.mean(fc[late_mask]) if late_mask.sum() else fc_mean
    else:
        fc_mid_mean  = fc_mean
        fc_late_mean = fc_mean

    # 최종 안정화 시점 추정 (FC 변동이 줄어드는 시점)
    window = min(20, len(fc)//4)
    if len(fc) > window*2:
        rolling_std = pd.Series(fc).rolling(window).std().values
        stable_idx  = np.argmin(rolling_std[window:]) + window
        fc_stable   = fc[stable_idx]
    else:
        fc_stable = fc_mean

    # 시험 지속시간
    duration = t[-1] - t[0]

    feat = {
        'name':          name,
        'fc_mean':       fc_mean,
        'fc_std':        fc_std,
        'fc_max':        fc_max,
        'fc_min':        fc_min,
        'fc_early_mean': fc_early_mean,
        'fc_early_max':  fc_early_max,
        'ramp_slope':    ramp_slope,
        'fc_mid_mean':   fc_mid_mean,
        'fc_late_mean':  fc_late_mean,
        'fc_stable':     fc_stable,
        'duration':      duration,
        'shutdown':      int(shutdown),
        'n_points':      len(df),
    }
    return feat

# ──────────────────────────────────────────
# 4. 데이터 로드 및 피처 추출
# ──────────────────────────────────────────
print("=" * 60)
print("CK26-12 예측 모델")
print("=" * 60)

dfs = {}
feats = {}
for key, path in FILES.items():
    dfs[key] = parse_tsv(path)
    feats[key] = extract_features(dfs[key], key)
    f = feats[key]
    print(f"\n[{key}] rows={f['n_points']}, duration={f['duration']:.0f}s, "
          f"shutdown={bool(f['shutdown'])}")
    print(f"  FC: mean={f['fc_mean']:.4f}, max={f['fc_max']:.4f}, "
          f"early_mean={f['fc_early_mean']:.4f}")
    print(f"  ramp_slope={f['ramp_slope']:.6f}, fc_stable={f['fc_stable']:.4f}")

# ──────────────────────────────────────────
# 5. 25→26 개선 비율 계산 (CK10, CK11 기준)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("25→26 개선 패턴 분석")
print("=" * 60)

pairs = [('25_ck10', '26_ck10'), ('25_ck11', '26_ck11')]
improvement = {}

for k25, k26 in pairs:
    f25 = feats[k25]
    f26 = feats[k26]
    ratio_mean       = f26['fc_mean']       / f25['fc_mean']
    ratio_max        = f26['fc_max']        / f25['fc_max']
    ratio_early      = f26['fc_early_mean'] / f25['fc_early_mean']
    ratio_ramp       = f26['ramp_slope']    / f25['ramp_slope']
    ratio_stable     = f26['fc_stable']     / f25['fc_stable']
    ratio_late       = f26['fc_late_mean']  / f25['fc_late_mean']

    print(f"\n{k25} → {k26}:")
    print(f"  fc_mean   비율: {ratio_mean:.4f}  ({(ratio_mean-1)*100:+.1f}%)")
    print(f"  fc_max    비율: {ratio_max:.4f}  ({(ratio_max-1)*100:+.1f}%)")
    print(f"  fc_early  비율: {ratio_early:.4f}  ({(ratio_early-1)*100:+.1f}%)")
    print(f"  ramp_slope비율: {ratio_ramp:.4f}  ({(ratio_ramp-1)*100:+.1f}%)")
    print(f"  fc_stable 비율: {ratio_stable:.4f}  ({(ratio_stable-1)*100:+.1f}%)")
    print(f"  fc_late   비율: {ratio_late:.4f}  ({(ratio_late-1)*100:+.1f}%)")

    improvement[k25] = {
        'ratio_mean':   ratio_mean,
        'ratio_max':    ratio_max,
        'ratio_early':  ratio_early,
        'ratio_ramp':   ratio_ramp,
        'ratio_stable': ratio_stable,
        'ratio_late':   ratio_late,
    }

# 평균 개선 비율
avg_ratio = {
    k: np.mean([improvement['25_ck10'][k], improvement['25_ck11'][k]])
    for k in improvement['25_ck10']
}
print(f"\n평균 개선 비율:")
for k, v in avg_ratio.items():
    print(f"  {k}: {v:.4f}  ({(v-1)*100:+.1f}%)")

# ──────────────────────────────────────────
# 6. CK26-12 마찰계수 예측
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("CK26-12 마찰계수 예측")
print("=" * 60)

f12 = feats['25_ck12']

# 방법 A: 25_CK12 피처 × 평균 개선비율
pred_fc_mean_A   = f12['fc_mean']       * avg_ratio['ratio_mean']
pred_fc_max_A    = f12['fc_max']        * avg_ratio['ratio_max']
pred_fc_early_A  = f12['fc_early_mean'] * avg_ratio['ratio_early']
pred_ramp_A      = f12['ramp_slope']    * avg_ratio['ratio_ramp']
pred_fc_stable_A = f12['fc_stable']     * avg_ratio['ratio_stable']
pred_fc_late_A   = f12['fc_late_mean']  * avg_ratio['ratio_late']

print("\n[방법 A: 평균 개선비율 적용]")
print(f"  예측 fc_mean   = {pred_fc_mean_A:.4f}")
print(f"  예측 fc_max    = {pred_fc_max_A:.4f}")
print(f"  예측 fc_early  = {pred_fc_early_A:.4f}")
print(f"  예측 ramp_slope= {pred_ramp_A:.6f}")
print(f"  예측 fc_stable = {pred_fc_stable_A:.4f}")
print(f"  예측 fc_late   = {pred_fc_late_A:.4f}")

# 25_CK12는 조기종료 → 실제 30분 시험시 안정화된 FC 추정 필요
# 안정화 예측: 초기 상승이 급격 → 안정화 후 fc_stable이 핵심
# 참고: 26_ck10 stable=0.228, 26_ck11 stable=0.271

# 방법 B: 선형 회귀 (26_CK10, 26_CK11 실제값 ← 25_CK10, 25_CK11 예측)
print("\n[방법 B: 실제 CK26-10/11 데이터와 25→26 선형 관계 모델링]")

# X = 25년도 피처, y = 26년도 fc_mean
x_train = np.array([feats['25_ck10']['fc_mean'], feats['25_ck11']['fc_mean']])
y_train = np.array([feats['26_ck10']['fc_mean'], feats['26_ck11']['fc_mean']])

# 단순 선형 회귀 (기울기 + 절편)
if len(x_train) >= 2:
    coeffs = np.polyfit(x_train, y_train, 1)
    slope_lr, intercept_lr = coeffs
    pred_fc_mean_B = slope_lr * f12['fc_mean'] + intercept_lr
    print(f"  선형모델: y = {slope_lr:.4f}x + {intercept_lr:.4f}")
    print(f"  예측 fc_mean = {pred_fc_mean_B:.4f}")

# fc_stable 기반 선형 회귀
x_stab = np.array([feats['25_ck10']['fc_stable'], feats['25_ck11']['fc_stable']])
y_stab = np.array([feats['26_ck10']['fc_stable'], feats['26_ck11']['fc_stable']])
coeffs_s = np.polyfit(x_stab, y_stab, 1)
pred_fc_stable_B = coeffs_s[0] * f12['fc_stable'] + coeffs_s[1]
print(f"  안정화 모델: y = {coeffs_s[0]:.4f}x + {coeffs_s[1]:.4f}")
print(f"  예측 fc_stable (안정화) = {pred_fc_stable_B:.4f}")

# ──────────────────────────────────────────
# 7. CK26-12 마모량 예측
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("CK26-12 마모량 예측")
print("=" * 60)

# 알려진 마모량
wear_25_ck10 = WEAR_AREA['25_ck10']  # 142020 μm²
wear_25_ck12 = WEAR_AREA['25_ck12']  # 189040 μm²

# 26년도 마모량은 아직 미확인이므로 FC 기반으로 추정
# 가정: 마모량 ∝ fc_mean (마찰력이 클수록 마모 심함)
# 25_CK10 vs 25_CK12: FC 비율로 마모 비율 추정
fc_ratio_12_to_10_25 = feats['25_ck12']['fc_mean'] / feats['25_ck10']['fc_mean']
wear_12_estimated_from_10 = wear_25_ck10 * (fc_ratio_12_to_10_25 ** 1.0)
print(f"\n25년 기준:")
print(f"  CK10 마모량={wear_25_ck10} μm², fc_mean={feats['25_ck10']['fc_mean']:.4f}")
print(f"  CK12 마모량={wear_25_ck12} μm², fc_mean={feats['25_ck12']['fc_mean']:.4f}")
print(f"  FC비율(CK12/CK10)={fc_ratio_12_to_10_25:.4f}")
print(f"  FC비율 기반 마모 추정치={wear_12_estimated_from_10:.0f} μm²")

# 26년도 CK12 마모량 예측
# 전략 1: 25_CK12 마모량 × 26/25 개선비율 (fc_mean 기준)
wear_improvement_ratio = avg_ratio['ratio_mean']
pred_wear_A = wear_25_ck12 * wear_improvement_ratio
print(f"\n[방법 A: 25_CK12 마모량 × FC 개선비율]")
print(f"  평균 FC 개선비율={wear_improvement_ratio:.4f}")
print(f"  예측 마모량={pred_wear_A:.0f} μm²")

# 전략 2: 25_CK10 마모량 기준, 예측 FC_mean 적용
pred_wear_B = wear_25_ck10 * (pred_fc_mean_A / feats['25_ck10']['fc_mean'])
print(f"\n[방법 B: 25_CK10 마모량 × 예측FC/기준FC]")
print(f"  예측 마모량={pred_wear_B:.0f} μm²")

# ──────────────────────────────────────────
# 8. 최종 예측값 요약
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 예측값 요약 — CK26-12")
print("=" * 60)

fc_final_A = pred_fc_mean_A
fc_final_B = pred_fc_mean_B if 'pred_fc_mean_B' in dir() else pred_fc_mean_A
fc_ensemble = np.mean([fc_final_A, fc_final_B])

wear_final_A = pred_wear_A
wear_final_B = pred_wear_B
wear_ensemble = np.mean([wear_final_A, wear_final_B])

print(f"\n마찰계수 (fc_mean):")
print(f"  방법 A (개선비율): {fc_final_A:.4f}")
print(f"  방법 B (선형회귀): {fc_final_B:.4f}")
print(f"  앙상블 평균:       {fc_ensemble:.4f}")

print(f"\n마찰계수 (fc_stable, 안정화):")
print(f"  방법 A: {pred_fc_stable_A:.4f}")
print(f"  방법 B: {pred_fc_stable_B:.4f}")
print(f"  앙상블: {np.mean([pred_fc_stable_A, pred_fc_stable_B]):.4f}")

print(f"\n마찰계수 (fc_max 예측):")
print(f"  방법 A: {pred_fc_max_A:.4f}")

print(f"\n마모량 (μm²):")
print(f"  방법 A: {wear_final_A:.0f}")
print(f"  방법 B: {wear_final_B:.0f}")
print(f"  앙상블: {wear_ensemble:.0f}")

print(f"\n참고 데이터:")
print(f"  25_CK10: fc_mean={feats['25_ck10']['fc_mean']:.4f}, wear={wear_25_ck10} μm²")
print(f"  25_CK11: fc_mean={feats['25_ck11']['fc_mean']:.4f}, wear={wear_25_ck12} μm²")
print(f"  25_CK12: fc_mean={feats['25_ck12']['fc_mean']:.4f}, wear={wear_25_ck12} μm²")
print(f"  26_CK10: fc_mean={feats['26_ck10']['fc_mean']:.4f}")
print(f"  26_CK11: fc_mean={feats['26_ck11']['fc_mean']:.4f}")

# ──────────────────────────────────────────
# 9. CK26-12 시계열 예측 (30분 전체)
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("CK26-12 마찰계수 시계열 시뮬레이션")
print("=" * 60)

# 26_CK10의 정규화된 시계열을 베이스로 스케일 조정
df_ref = dfs['26_ck10'].copy()
t_ref = df_ref['Test Time'].values
fc_ref = df_ref['Friction Coefficient'].values

# CK26-12 예측: CK26-10 시계열 × (예측 fc_mean / 26_CK10 fc_mean)
scale = fc_ensemble / feats['26_ck10']['fc_mean']
fc_pred_series = fc_ref * scale

print(f"\nCK26-10 기준 스케일 팩터: {scale:.4f}")
print(f"예측 시계열 통계:")
print(f"  mean={np.mean(fc_pred_series):.4f}, max={np.max(fc_pred_series):.4f}, "
      f"min={np.min(fc_pred_series):.4f}, std={np.std(fc_pred_series):.4f}")

# 시계열 주요 구간 출력
checkpoints = [0, 5, 10, 15, 20, 25, 30]  # 분
print(f"\n시간대별 예측 마찰계수:")
for min_target in checkpoints:
    sec_target = min_target * 60
    idx = np.argmin(np.abs(t_ref - sec_target))
    if idx < len(fc_pred_series):
        print(f"  {min_target:2d}분 ({t_ref[idx]:.0f}s): FC={fc_pred_series[idx]:.4f}")

print("\n" + "=" * 60)
print("예측 완료. 실제 CK26-12 데이터를 제공하면 정확도를 검증하겠습니다.")
print("=" * 60)
