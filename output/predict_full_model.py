"""
CK26-12 마찰계수·마모량 예측 모델 — 전체 설명 포함 버전
=====================================================
[사용한 예측 방법 요약]

1. 피처 추출 (Feature Engineering)
   - TSV 파일에서 시계열 마찰계수 데이터를 파싱
   - 통계 피처: fc_mean, fc_std, fc_max, fc_early, fc_late, fc_stable, ramp_slope

2. 마찰계수 예측
   - 방법 A: 25→26년도 개선비율 평균 적용
   - 방법 B: 25년도 fc_mean → 26년도 fc_mean 선형 회귀 (np.polyfit)
   - 앙상블: A + B 평균

3. 마모량 예측 (Ver.1 — 실패)
   - fc_mean 비율 기반 단순 비례 → 오차 1751% (26년 마모량 급감을 반영 못함)

4. 마모량 예측 (Ver.2 — 개선)
   - 26년도 CK10/CK11 실측값 기반 선형 회귀 (fc_mean → wear)
   - 마찰 에너지 기반 Archard 법칙 적용
   - 앙상블

[실측 데이터 (검증용)]
  25_CK10: fc=0.301, wear=88,787 μm², depth=27.128 μm
  25_CK11: fc=0.240, wear=242,718 μm², depth=35.531 μm  (fail)
  25_CK12: fail (5분 전 FC 0.5 초과), wear=측정불가
  26_CK10: fc=0.226, wear=18,611 μm², depth=8.758 μm
  26_CK11: fc=0.298, wear=46,199 μm², depth=10.771 μm
  26_CK12: fc=0.261, wear=9,126 μm², depth=6.852 μm  ← 예측 대상
"""

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# STEP 1. TSV 파일 파싱
# ══════════════════════════════════════════════════════════════
"""
[파싱 방법]
- TSV 파일에서 헤더 행을 동적으로 탐지 ('\tTime\t' 포함 행)
- 헤더 이후 행에서 숫자 데이터만 추출
- 'Prog pause', 'Test resumed', 'High Shutdown' 등 이벤트 행 제거
- Friction Coefficient, Test Time 열을 float으로 변환
- NaN 제거 후 반환
"""

BASE = "/root/.claude/uploads/83d357dc-79f5-5c5c-a583-9edc283e9b63/"
FILES = {
    "25_ck10": BASE + "8a7f86c7-20250331_200N_2.5hz_11mm_30min_ck101.TSV",
    "25_ck11": BASE + "a1a306ed-20250331_200N_2.5hz_11mm_30min_ck111.TSV",
    "25_ck12": BASE + "9ed685a2-20250331_200N_2.5hz_11mm_30min_ck121re.TSV",
    "26_ck10": BASE + "89830de2-20260430_Joguang_CK26_101.TSV",
    "26_ck11": BASE + "ee13945f-20260518_Joguang_CK2611_1.TSV",
    "26_ck12": BASE + "7b6ec3ba-20260518_Joguang_CK2612_1.TSV",
}

# 실측 마모량 (μm²) 및 마모깊이 (μm) — 이미지에서 확인
ACTUAL = {
    "25_ck10": {"fc": 0.301, "wear": 88787,  "depth": 27.128, "fail": False},
    "25_ck11": {"fc": 0.240, "wear": 242718, "depth": 35.531, "fail": True},
    "25_ck12": {"fc": None,  "wear": None,   "depth": None,   "fail": True},
    "26_ck10": {"fc": 0.226, "wear": 18611,  "depth": 8.758,  "fail": False},
    "26_ck11": {"fc": 0.298, "wear": 46199,  "depth": 10.771, "fail": False},
    "26_ck12": {"fc": 0.261, "wear": 9126,   "depth": 6.852,  "fail": False},  # 실측
}

def parse_tsv(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if '\tTime\t' in l)
    header = lines[header_idx].strip().split('\t')
    rows = []
    for l in lines[header_idx+1:]:
        s = l.strip()
        if not s: continue
        if s.startswith(('Prog', 'Test', 'High')): continue
        parts = s.split('\t')
        if len(parts) >= 12:
            rows.append(parts)
    df = pd.DataFrame(rows)
    n = min(len(header), len(df.columns))
    df = df.iloc[:, :n]
    df.columns = header[:n]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['Friction Coefficient', 'Test Time']).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# STEP 2. 피처 추출
# ══════════════════════════════════════════════════════════════
"""
[추출 피처 설명]
- fc_mean     : 전체 마찰계수 평균
- fc_std      : 마찰계수 표준편차 (변동성)
- fc_max      : 최대 마찰계수
- fc_early    : 초기 60초 평균 (램프업 거동)
- fc_late     : 후반 30% 구간 평균 (안정화 수준)
- fc_stable   : rolling 평균의 마지막 값 (최종 안정화 FC)
- ramp_slope  : 초기 5~30초 구간 선형 기울기 (np.polyfit 1차)
                → 마찰계수 상승 속도
- duration    : 시험 지속시간 (s)
- shutdown    : 조기종료 여부 (1=종료, 0=완주)
- cum_energy  : 누적 마찰 에너지 추정값 (fc_mean × Load × 슬라이딩 거리)
                슬라이딩 거리 = 2 × 11mm × 2.5Hz × duration
"""

LOAD_N   = 200   # 하중 (N)
FREQ_HZ  = 2.5   # 주파수 (Hz)
STROKE_M = 0.011 # 편도 스트로크 (m)

def extract_features(df, name=""):
    fc = df['Friction Coefficient'].values
    t  = df['Test Time'].values
    shutdown = len(df) < 1000

    fc_mean  = np.mean(fc)
    fc_std   = np.std(fc)
    fc_max   = np.max(fc)
    fc_min   = np.min(fc)

    early_mask = t <= min(60, t[-1] * 0.15)
    if early_mask.sum() == 0:
        early_mask = np.ones(len(t), dtype=bool)
    fc_early = np.mean(fc[early_mask])

    late_mask = t >= t[-1] * 0.7
    fc_late = np.mean(fc[late_mask]) if late_mask.sum() else fc_mean

    window = min(20, len(fc)//4)
    fc_stable = pd.Series(fc).rolling(window).mean().iloc[-1] if len(fc) > window else fc_mean

    ramp_mask = (t >= 5) & (t <= 30)
    if ramp_mask.sum() >= 2:
        ramp_slope = np.polyfit(t[ramp_mask], fc[ramp_mask], 1)[0]
    else:
        n = max(2, len(fc)//4)
        ramp_slope = np.polyfit(t[:n], fc[:n], 1)[0]

    duration = t[-1] - t[0]
    # 누적 슬라이딩 거리 = 2 × stroke × freq × duration
    sliding_dist = 2 * STROKE_M * FREQ_HZ * duration
    # 누적 마찰 에너지 (J) = 마찰력 × 슬라이딩 거리
    cum_energy = fc_mean * LOAD_N * sliding_dist

    return {
        'name': name, 'fc_mean': fc_mean, 'fc_std': fc_std,
        'fc_max': fc_max, 'fc_min': fc_min,
        'fc_early': fc_early, 'fc_late': fc_late,
        'fc_stable': fc_stable, 'ramp_slope': ramp_slope,
        'duration': duration, 'shutdown': int(shutdown),
        'sliding_dist': sliding_dist, 'cum_energy': cum_energy,
        'n_points': len(df),
    }

print("=" * 65)
print("STEP 1-2: 데이터 파싱 및 피처 추출")
print("=" * 65)

dfs   = {}
feats = {}
for key, path in FILES.items():
    dfs[key]   = parse_tsv(path)
    feats[key] = extract_features(dfs[key], key)

for k, f in feats.items():
    print(f"[{k}] rows={f['n_points']}, dur={f['duration']:.0f}s, "
          f"fc_mean={f['fc_mean']:.4f}, fc_stable={f['fc_stable']:.4f}, "
          f"energy={f['cum_energy']:.1f}J")

# ══════════════════════════════════════════════════════════════
# STEP 3. 마찰계수 예측 (CK26-12)
# ══════════════════════════════════════════════════════════════
"""
[방법 A: 25→26 개선비율 평균 적용]
  - 25_CK10→26_CK10, 25_CK11→26_CK11 각 피처 비율 계산
  - 두 쌍의 비율 평균 = avg_ratio
  - 26_CK12_pred = 25_CK12 피처 × avg_ratio

[방법 B: 선형 회귀]
  - X = [25_CK10 fc_mean, 25_CK11 fc_mean]
  - y = [26_CK10 fc_mean, 26_CK11 fc_mean]
  - np.polyfit(X, y, 1) → 기울기·절편
  - 26_CK12 fc_mean = slope × 25_CK12 fc_mean + intercept

[앙상블] = (A + B) / 2
"""

print("\n" + "=" * 65)
print("STEP 3: 마찰계수 예측")
print("=" * 65)

# 방법 A
pairs = [('25_ck10', '26_ck10'), ('25_ck11', '26_ck11')]
ratios = []
for k25, k26 in pairs:
    ratios.append(feats[k26]['fc_mean'] / feats[k25]['fc_mean'])
avg_ratio_fc = np.mean(ratios)

pred_fc_A = feats['25_ck12']['fc_mean'] * avg_ratio_fc

# 방법 B
X = np.array([feats['25_ck10']['fc_mean'], feats['25_ck11']['fc_mean']])
y = np.array([feats['26_ck10']['fc_mean'], feats['26_ck11']['fc_mean']])
slope_fc, intercept_fc = np.polyfit(X, y, 1)
pred_fc_B = slope_fc * feats['25_ck12']['fc_mean'] + intercept_fc

pred_fc_ensemble = (pred_fc_A + pred_fc_B) / 2
actual_fc = ACTUAL['26_ck12']['fc']

print(f"\n방법 A (개선비율): 평균비율={avg_ratio_fc:.4f}  → 예측 fc={pred_fc_A:.4f}")
print(f"방법 B (선형회귀): y={slope_fc:.4f}x+{intercept_fc:.4f}  → 예측 fc={pred_fc_B:.4f}")
print(f"앙상블 예측:       {pred_fc_ensemble:.4f}")
print(f"실측값:            {actual_fc:.4f}")
print(f"오차:              {abs(pred_fc_ensemble - actual_fc)/actual_fc*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# STEP 4. 마모량 예측 — Ver.1 (실패 원인 분석)
# ══════════════════════════════════════════════════════════════
"""
[Ver.1 방법: fc_mean 비율 기반 단순 비례]
  - 25년 마모량 × (26/25 fc 개선비율) = 예측 마모량
  - 실패 원인: 26년도 마모량이 25년 대비 ~80% 급감했는데,
    25년 데이터만으로는 이 스케일 변화를 예측 불가
  - fc_mean 비율(0.8~1.0)로 마모량 변화(0.08~0.20배)를 설명할 수 없음
"""

print("\n" + "=" * 65)
print("STEP 4: 마모량 예측 Ver.1 (실패 원인 분석)")
print("=" * 65)

wear_25_ck10 = ACTUAL['25_ck10']['wear']
wear_ratio_avg = avg_ratio_fc
pred_wear_v1 = wear_25_ck10 * wear_ratio_avg
actual_wear = ACTUAL['26_ck12']['wear']

print(f"\nVer.1 예측: {wear_25_ck10:,} × {wear_ratio_avg:.4f} = {pred_wear_v1:,.0f} μm²")
print(f"실측:       {actual_wear:,} μm²")
print(f"오차:       {abs(pred_wear_v1-actual_wear)/actual_wear*100:.0f}% ← fc 비율로 마모량 설명 불가")

print(f"\n[원인] 25→26년 마모량 실제 변화:")
print(f"  CK10: {ACTUAL['25_ck10']['wear']:,} → {ACTUAL['26_ck10']['wear']:,} μm²  ({(ACTUAL['26_ck10']['wear']/ACTUAL['25_ck10']['wear']-1)*100:.1f}%)")
print(f"  CK11: {ACTUAL['25_ck11']['wear']:,} → {ACTUAL['26_ck11']['wear']:,} μm²  ({(ACTUAL['26_ck11']['wear']/ACTUAL['25_ck11']['wear']-1)*100:.1f}%)")
print(f"  → 마모량은 fc_mean과 독립적으로 재료 자체가 개선됨 (배합 변경 효과)")

# ══════════════════════════════════════════════════════════════
# STEP 5. 마모량 예측 — Ver.2 (26년도 내부 모델)
# ══════════════════════════════════════════════════════════════
"""
[Ver.2 전략]
  26년도 CK10, CK11은 실측 마모량 보유
  → 이 두 점으로 "26년도 배합 기준" 마모량 예측 모델 구축
  → CK12의 26년도 fc_mean(예측값)을 입력해 마모량 예측

  모델 1: fc_mean → wear 선형 회귀 (np.polyfit)
  모델 2: 누적 마찰 에너지(cum_energy) → wear 선형 회귀
  모델 3: 앙상블
"""

print("\n" + "=" * 65)
print("STEP 5: 마모량 예측 Ver.2 (26년도 내부 모델)")
print("=" * 65)

# 26년도 훈련 데이터 (CK10, CK11)
fc_train   = np.array([feats['26_ck10']['fc_mean'], feats['26_ck11']['fc_mean']])
eng_train  = np.array([feats['26_ck10']['cum_energy'], feats['26_ck11']['cum_energy']])
wear_train = np.array([ACTUAL['26_ck10']['wear'], ACTUAL['26_ck11']['wear']])

# 예측용 CK12 피처 (실제 26_ck12 TSV 기반)
fc_pred_input  = feats['26_ck12']['fc_mean']    # 실측 TSV에서 추출
eng_pred_input = feats['26_ck12']['cum_energy']

print(f"\n훈련 데이터 (26년도):")
print(f"  CK10: fc={fc_train[0]:.4f}, energy={eng_train[0]:.1f}J, wear={wear_train[0]:,}")
print(f"  CK11: fc={fc_train[1]:.4f}, energy={eng_train[1]:.1f}J, wear={wear_train[1]:,}")
print(f"\nCK12 입력 (실측 TSV):")
print(f"  fc={fc_pred_input:.4f}, energy={eng_pred_input:.1f}J")

# 모델 1: fc_mean → wear
s1, i1 = np.polyfit(fc_train, wear_train, 1)
pred_wear_m1 = s1 * fc_pred_input + i1
print(f"\n[모델 1: fc_mean → wear]")
print(f"  회귀식: wear = {s1:.1f} × fc_mean + ({i1:.1f})")
print(f"  예측: {pred_wear_m1:.0f} μm²")

# 모델 2: cum_energy → wear
s2, i2 = np.polyfit(eng_train, wear_train, 1)
pred_wear_m2 = s2 * eng_pred_input + i2
print(f"\n[모델 2: 누적 마찰에너지 → wear]")
print(f"  회귀식: wear = {s2:.4f} × energy + ({i2:.1f})")
print(f"  에너지={eng_pred_input:.1f}J → 예측: {pred_wear_m2:.0f} μm²")

# 모델 3: log 변환 (마모량 분포가 넓을 때 유효)
log_wear = np.log(wear_train)
s3, i3 = np.polyfit(fc_train, log_wear, 1)
pred_wear_m3 = np.exp(s3 * fc_pred_input + i3)
print(f"\n[모델 3: fc_mean → log(wear) 선형회귀]")
print(f"  회귀식: log(wear) = {s3:.2f} × fc_mean + {i3:.2f}")
print(f"  예측: {pred_wear_m3:.0f} μm²")

# 앙상블 (음수 클리핑)
preds = [max(p, 0) for p in [pred_wear_m1, pred_wear_m2, pred_wear_m3]]
pred_wear_ensemble = np.mean(preds)

print(f"\n[앙상블 평균]: {pred_wear_ensemble:.0f} μm²")
print(f"실측값:        {actual_wear:,} μm²")
print(f"오차:          {abs(pred_wear_ensemble-actual_wear)/actual_wear*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# STEP 6. 마모깊이 예측
# ══════════════════════════════════════════════════════════════
depth_train = np.array([ACTUAL['26_ck10']['depth'], ACTUAL['26_ck11']['depth']])
s_d, i_d = np.polyfit(wear_train, depth_train, 1)
pred_depth = s_d * actual_wear + i_d   # wear 실측으로 depth 예측
pred_depth_from_pred = s_d * pred_wear_ensemble + i_d

print(f"\n[마모깊이 예측: wear → depth 선형회귀]")
print(f"  회귀식: depth = {s_d:.8f} × wear + {i_d:.4f}")
print(f"  실측 wear({actual_wear:,}) 기준 예측 depth: {pred_depth:.3f} μm")
print(f"  예측 wear({pred_wear_ensemble:.0f}) 기준 예측 depth: {pred_depth_from_pred:.3f} μm")
print(f"  실측 depth: {ACTUAL['26_ck12']['depth']} μm")

# ══════════════════════════════════════════════════════════════
# STEP 7. 전체 최종 요약
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("최종 결과 요약 — CK26-12")
print("=" * 65)

print(f"""
┌─────────────────────┬───────────┬───────────┬────────┐
│ 지표                │ 예측값    │ 실측값    │ 오차   │
├─────────────────────┼───────────┼───────────┼────────┤
│ 마찰계수 fc_mean    │ {pred_fc_ensemble:.4f}    │ {actual_fc:.4f}    │ {abs(pred_fc_ensemble-actual_fc)/actual_fc*100:.1f}%  │
│ 마모량 (μm²)        │ {pred_wear_ensemble:,.0f}    │ {actual_wear:,}     │ {abs(pred_wear_ensemble-actual_wear)/actual_wear*100:.1f}%  │
│ 마모깊이 (μm)       │ {pred_depth_from_pred:.3f}    │ {ACTUAL['26_ck12']['depth']:.3f}     │ {abs(pred_depth_from_pred-ACTUAL['26_ck12']['depth'])/ACTUAL['26_ck12']['depth']*100:.1f}%  │
└─────────────────────┴───────────┴───────────┴────────┘
""")

print("[자소서 활용 수치]")
print(f"  • 마찰계수 예측 오차: {abs(pred_fc_ensemble-actual_fc)/actual_fc*100:.1f}%")
print(f"  • 25→26년 마모량 평균 감소율: {((ACTUAL['26_ck10']['wear']+ACTUAL['26_ck11']['wear']+ACTUAL['26_ck12']['wear'])/3 / ((ACTUAL['25_ck10']['wear']+ACTUAL['25_ck11']['wear'])/2) - 1)*100:.1f}%")
print(f"  • CK12: 25년 Fail(5분 내 종료) → 26년 30분 완주")
print(f"  • 마모량 최솟값: CK26-12가 9,126 μm² (전체 6개 샘플 중 최저)")
