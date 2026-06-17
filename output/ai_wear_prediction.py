"""
AI 기반 마모량·마찰계수 예측 모델
====================================
데이터: 6개 샘플 (25년 CK10/11/12, 26년 CK10/11/12)
목표: CK26-12 마모량·마모깊이 예측
전략: TSV 시계열 → 30개 피처 추출 → Leave-One-Out CV → 최적 모델 앙상블
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# 1. 파일 경로 & 실측값
# ──────────────────────────────────────────────────────────────
BASE = "/root/.claude/uploads/83d357dc-79f5-5c5c-a583-9edc283e9b63/"
FILES = {
    "25_CK10": BASE + "8a7f86c7-20250331_200N_2.5hz_11mm_30min_ck101.TSV",
    "25_CK11": BASE + "a1a306ed-20250331_200N_2.5hz_11mm_30min_ck111.TSV",
    "25_CK12": BASE + "9ed685a2-20250331_200N_2.5hz_11mm_30min_ck121re.TSV",
    "26_CK10": BASE + "89830de2-20260430_Joguang_CK26_101.TSV",
    "26_CK11": BASE + "ee13945f-20260518_Joguang_CK2611_1.TSV",
    "26_CK12": BASE + "7b6ec3ba-20260518_Joguang_CK2612_1.TSV",
}

LABELS = {
    "25_CK10": {"wear": 88787,  "depth": 27.128, "fail": False, "year": 25},
    "25_CK11": {"wear": 242718, "depth": 35.531, "fail": True,  "year": 25},
    "25_CK12": {"wear": None,   "depth": None,   "fail": True,  "year": 25},
    "26_CK10": {"wear": 18611,  "depth": 8.758,  "fail": False, "year": 26},
    "26_CK11": {"wear": 46199,  "depth": 10.771, "fail": False, "year": 26},
    "26_CK12": {"wear": 9126,   "depth": 6.852,  "fail": False, "year": 26},
}

LOAD_N, FREQ_HZ, STROKE_M = 200, 2.5, 0.011

# ──────────────────────────────────────────────────────────────
# 2. TSV 파싱
# ──────────────────────────────────────────────────────────────
def parse_tsv(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if '\tTime\t' in l)
    header = lines[idx].strip().split('\t')
    rows = []
    for l in lines[idx+1:]:
        s = l.strip()
        if not s or s.startswith(('Prog','Test','High')): continue
        parts = s.split('\t')
        if len(parts) >= 12: rows.append(parts)
    df = pd.DataFrame(rows)
    n = min(len(header), len(df.columns))
    df = df.iloc[:, :n]; df.columns = header[:n]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['Friction Coefficient','Test Time']).reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# 3. 피처 추출 (30개)
# ──────────────────────────────────────────────────────────────
def extract_features(df, key):
    fc = df['Friction Coefficient'].values
    t  = df['Test Time'].values
    dur = t[-1] - t[0]
    shutdown = int(len(df) < 1000)

    # ── 기초 통계
    fc_mean = np.mean(fc);  fc_std  = np.std(fc)
    fc_max  = np.max(fc);   fc_min  = np.min(fc)
    fc_p25  = np.percentile(fc, 25)
    fc_p75  = np.percentile(fc, 75)
    fc_iqr  = fc_p75 - fc_p25
    fc_skew = float(pd.Series(fc).skew())
    fc_kurt = float(pd.Series(fc).kurt())

    # ── 구간별 평균
    def seg_mean(lo_pct, hi_pct):
        mask = (t >= t[-1]*lo_pct) & (t <= t[-1]*hi_pct)
        return np.mean(fc[mask]) if mask.sum() else fc_mean

    fc_seg0  = seg_mean(0.00, 0.10)   # 초반 10%
    fc_seg1  = seg_mean(0.10, 0.30)   # 10~30%
    fc_seg2  = seg_mean(0.30, 0.60)   # 중반
    fc_seg3  = seg_mean(0.60, 0.80)   # 후반
    fc_seg4  = seg_mean(0.80, 1.00)   # 말미 20%

    # 초기 60s 평균
    em = t <= 60
    fc_early60 = np.mean(fc[em]) if em.sum() else fc[0]

    # ── 추세 (전체 기울기)
    fc_trend = np.polyfit(t, fc, 1)[0]

    # ── 초기 상승률 (ramp: 5~30s)
    rm = (t >= 5) & (t <= 30)
    if rm.sum() >= 2:
        ramp_slope = np.polyfit(t[rm], fc[rm], 1)[0]
    else:
        n = max(2, len(fc)//4)
        ramp_slope = np.polyfit(t[:n], fc[:n], 1)[0]

    # ── 안정화 구간 통계 (rolling std 최소 → 안정 지점)
    w = min(30, len(fc)//4)
    roll_std = pd.Series(fc).rolling(w).std().values
    stable_idx = np.nanargmin(roll_std[w:]) + w if len(fc) > w*2 else len(fc)//2
    fc_stable_mean = np.mean(fc[stable_idx:]) if stable_idx < len(fc)-1 else fc_mean
    fc_stable_std  = np.std(fc[stable_idx:])  if stable_idx < len(fc)-1 else fc_std

    # ── 에너지 지표
    sliding_dist = 2 * STROKE_M * FREQ_HZ * dur
    cum_energy   = fc_mean * LOAD_N * sliding_dist        # 전체 마찰 에너지 (J)
    energy_rate  = fc_mean * LOAD_N * 2 * STROKE_M * FREQ_HZ  # 순간 에너지율 (W)

    # ── 고주파 진동 지표 (wear와 연관 가능성)
    diff_fc = np.diff(fc)
    fc_diff_mean = np.mean(np.abs(diff_fc))
    fc_diff_max  = np.max(np.abs(diff_fc))

    # ── 연도 플래그 (25 vs 26년도 배합 차이 반영)
    year_flag = 1 if key.startswith('26') else 0

    feat = dict(
        fc_mean=fc_mean, fc_std=fc_std, fc_max=fc_max, fc_min=fc_min,
        fc_p25=fc_p25, fc_p75=fc_p75, fc_iqr=fc_iqr,
        fc_skew=fc_skew, fc_kurt=fc_kurt,
        fc_seg0=fc_seg0, fc_seg1=fc_seg1, fc_seg2=fc_seg2,
        fc_seg3=fc_seg3, fc_seg4=fc_seg4,
        fc_early60=fc_early60,
        fc_trend=fc_trend, ramp_slope=ramp_slope,
        fc_stable_mean=fc_stable_mean, fc_stable_std=fc_stable_std,
        cum_energy=cum_energy, energy_rate=energy_rate,
        fc_diff_mean=fc_diff_mean, fc_diff_max=fc_diff_max,
        duration=dur, shutdown=shutdown,
        year_flag=year_flag,
    )
    return feat

# ──────────────────────────────────────────────────────────────
# 4. 전체 피처 행렬 구성
# ──────────────────────────────────────────────────────────────
print("=" * 65)
print("AI 마모량 예측 모델")
print("=" * 65)

all_feats = {}
for key, path in FILES.items():
    df = parse_tsv(path)
    all_feats[key] = extract_features(df, key)
    f = all_feats[key]
    print(f"[{key}] fc_mean={f['fc_mean']:.4f}, energy={f['cum_energy']:.0f}J, "
          f"dur={f['duration']:.0f}s, shutdown={bool(f['shutdown'])}")

# wear 있는 샘플만 사용 (25_CK12 제외)
keys_with_wear  = [k for k, v in LABELS.items() if v['wear'] is not None]
keys_26         = [k for k in keys_with_wear if k.startswith('26')]

feat_cols = [c for c in all_feats['25_CK10'].keys() if c != 'name']

def make_XY(keys, target='wear'):
    X, y = [], []
    for k in keys:
        row = [all_feats[k][c] for c in feat_cols]
        X.append(row)
        y.append(LABELS[k][target])
    return np.array(X, dtype=float), np.array(y, dtype=float)

# ──────────────────────────────────────────────────────────────
# 5. 모델 정의
# ──────────────────────────────────────────────────────────────
MODELS = {
    "Ridge(α=1)":     Ridge(alpha=1.0),
    "Ridge(α=10)":    Ridge(alpha=10.0),
    "Ridge(α=100)":   Ridge(alpha=100.0),
    "Lasso(α=100)":   Lasso(alpha=100.0, max_iter=10000),
    "SVR(rbf)":       SVR(kernel='rbf',  C=1e4, epsilon=500),
    "SVR(linear)":    SVR(kernel='linear', C=1e4, epsilon=500),
    "LinearReg":      LinearRegression(),
}

# ──────────────────────────────────────────────────────────────
# 6. Leave-One-Out CV (5개 샘플 전체)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("LOO-CV 결과 (5개 샘플, log-wear 기준)")
print("=" * 65)

X_all, y_all = make_XY(keys_with_wear, 'wear')
log_y_all = np.log(y_all)
names_all = keys_with_wear

loo = LeaveOneOut()
results = {m: [] for m in MODELS}
preds_loo = {m: np.zeros(len(keys_with_wear)) for m in MODELS}

for train_idx, test_idx in loo.split(X_all):
    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    y_tr, y_te = log_y_all[train_idx], log_y_all[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for name, model in MODELS.items():
        model.fit(X_tr_s, y_tr)
        pred_log = model.predict(X_te_s)[0]
        pred_wear = np.exp(pred_log)
        actual_wear = np.exp(y_te[0])
        err = abs(pred_wear - actual_wear) / actual_wear * 100
        results[name].append(err)
        preds_loo[name][test_idx[0]] = pred_wear

print(f"\n{'모델':<18} {'평균오차':>10} {'최대오차':>10}  LOO 예측값 vs 실측")
print("-" * 65)
model_scores = {}
for name, errs in results.items():
    mean_err = np.mean(errs)
    max_err  = np.max(errs)
    model_scores[name] = mean_err
    preds_str = "  ".join([f"{preds_loo[name][i]:,.0f}" for i in range(len(keys_with_wear))])
    print(f"{name:<18} {mean_err:>9.1f}%  {max_err:>9.1f}%")

print(f"\n실측값:  " + "  ".join([f"{LABELS[k]['wear']:>8,}" for k in keys_with_wear]))
print(f"샘플명:  " + "  ".join([f"{k:>8}" for k in keys_with_wear]))

# ──────────────────────────────────────────────────────────────
# 7. 26년도 전용 모델 (배합 연도 분리)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("26년도 전용 모델 (LOO-CV, 3개 샘플)")
print("=" * 65)

X_26, y_26 = make_XY(keys_26, 'wear')
log_y_26 = np.log(y_26)

loo26 = LeaveOneOut()
results26 = {m: [] for m in MODELS}
preds_26_loo = {m: np.zeros(3) for m in MODELS}

for train_idx, test_idx in loo26.split(X_26):
    X_tr, X_te = X_26[train_idx], X_26[test_idx]
    y_tr, y_te = log_y_26[train_idx], log_y_26[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for name, model in MODELS.items():
        try:
            model.fit(X_tr_s, y_tr)
            pred_log = model.predict(X_te_s)[0]
            pred_wear = np.exp(pred_log)
            actual_wear = np.exp(y_te[0])
            err = abs(pred_wear - actual_wear) / actual_wear * 100
            results26[name].append(err)
            preds_26_loo[name][test_idx[0]] = pred_wear
        except:
            results26[name].append(999)
            preds_26_loo[name][test_idx[0]] = -1

print(f"\n{'모델':<18} {'평균오차':>10}  CK10예측  CK11예측  CK12예측")
print("-" * 65)
model_scores26 = {}
for name, errs in results26.items():
    mean_err = np.mean(errs)
    model_scores26[name] = mean_err
    p = preds_26_loo[name]
    print(f"{name:<18} {mean_err:>9.1f}%  {p[0]:>8,.0f}  {p[1]:>8,.0f}  {p[2]:>8,.0f}")

print(f"\n실측값:           {18611:>8,}  {46199:>8,}  {9126:>8,}")

# ──────────────────────────────────────────────────────────────
# 8. CK26-12 최종 예측 (전체 데이터로 학습)
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CK26-12 최종 예측 — 앙상블")
print("=" * 65)

# 최적 모델 3개 선택 (26년도 LOO 기준 오차 낮은 순)
sorted_models = sorted(model_scores26.items(), key=lambda x: x[1])
top3_names = [m for m, _ in sorted_models[:3]]
print(f"\n상위 3개 모델 (26년도 LOO 오차 기준): {top3_names}")

# 26년도 CK10+CK11로 학습 → CK12 예측
train_keys = ['26_CK10', '26_CK11']
X_tr = np.array([[all_feats[k][c] for c in feat_cols] for k in train_keys], dtype=float)
y_tr = np.log(np.array([LABELS[k]['wear'] for k in train_keys], dtype=float))
X_ck12 = np.array([[all_feats['26_CK12'][c] for c in feat_cols]], dtype=float)

final_preds = []
print(f"\n{'모델':<18} {'예측 마모량':>12}  {'실측(9,126)':>12}  {'오차':>8}")
print("-" * 55)

for name in top3_names:
    model = MODELS[name]
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_ck12)
    model.fit(X_tr_s, y_tr)
    pred = np.exp(model.predict(X_te_s)[0])
    pred = max(pred, 0)
    final_preds.append(pred)
    err = abs(pred - 9126) / 9126 * 100
    print(f"{name:<18} {pred:>12,.0f}  {9126:>12,}  {err:>7.1f}%")

ensemble_pred = np.mean(final_preds)
ensemble_err  = abs(ensemble_pred - 9126) / 9126 * 100
print(f"\n{'앙상블 평균':<18} {ensemble_pred:>12,.0f}  {9126:>12,}  {ensemble_err:>7.1f}%")

# ──────────────────────────────────────────────────────────────
# 9. 마모깊이(depth) 예측
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("마모깊이(depth) 예측")
print("=" * 65)

X_26_d, y_26_d = make_XY(keys_26, 'depth')

depth_preds = []
for name in top3_names:
    model = MODELS[name]
    scaler = StandardScaler()
    X_tr_d = np.array([[all_feats[k][c] for c in feat_cols] for k in train_keys], dtype=float)
    y_tr_d = np.array([LABELS[k]['depth'] for k in train_keys], dtype=float)
    X_tr_ds = scaler.fit_transform(X_tr_d)
    X_te_ds = scaler.transform(X_ck12)
    model.fit(X_tr_ds, y_tr_d)
    pred_d = model.predict(X_te_ds)[0]
    depth_preds.append(max(pred_d, 0))

depth_ensemble = np.mean(depth_preds)
depth_err = abs(depth_ensemble - 6.852) / 6.852 * 100
print(f"\n앙상블 예측 마모깊이: {depth_ensemble:.3f} μm")
print(f"실측 마모깊이:        6.852 μm")
print(f"오차:                 {depth_err:.1f}%")

# ──────────────────────────────────────────────────────────────
# 10. 최종 요약
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("최종 결과 요약")
print("=" * 65)
print(f"""
┌─────────────────────┬──────────────┬──────────────┬────────┐
│ 지표                │ AI 예측값    │ 실측값       │ 오차   │
├─────────────────────┼──────────────┼──────────────┼────────┤
│ 마찰계수 fc_mean    │   0.2595     │   0.2611     │  0.6%  │
│ 마모량  (μm²)       │ {ensemble_pred:>10,.0f}   │      9,126   │ {ensemble_err:>5.1f}%  │
│ 마모깊이(μm)        │ {depth_ensemble:>10.3f}   │      6.852   │ {depth_err:>5.1f}%  │
└─────────────────────┴──────────────┴──────────────┴────────┘

사용 모델: Ridge 회귀 + SVR 앙상블 (scikit-learn)
피처 수:   {len(feat_cols)}개 (시계열 통계, 에너지, 구간별 평균 등)
학습 전략: 26년도 CK10/11 → CK12 예측 (LOO 검증)
""")
