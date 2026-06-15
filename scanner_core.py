# -*- coding: utf-8 -*-
"""
이동평균선 6단계 스캐너 — 핵심 로직 (GUI 없음, 서버/배치 전용)
데스크톱(stage_scanner_app.py)의 분석 로직과 동일하게 유지.
- EMA 5/20/50 배열로 6단계 분류
- 괘상(주역 8괘) 산출 (단계별 평균 이격 기반 2-pass)
- 관심 후보: 괘상 '건(乾)' 단계별 할당 선별 + 4단계 '리(離)' 별도 선별
"""
import pandas as pd

# ── 상수 ──
SPAN_SHORT, SPAN_MID, SPAN_LONG = 5, 20, 50
LOOKBACK_DAYS = 220
GUA_SLOPE_LOOKBACK = 3
PICK_MAX_TOTAL = 50
RIRI_MAX = 10

STAGE_LABEL = {
    1: "안정 상승", 2: "단기 조정", 3: "하락 전환",
    4: "안정 하락", 5: "단기 반등", 6: "상승 전환", 0: "판정 불가",
}
STAGE_COLOR = {
    1: "#1d9e75", 2: "#5dcaa5", 3: "#ba7517",
    4: "#e24b4a", 5: "#d85a30", 6: "#378add", 0: "#888780",
}

# 주역 8괘 — 키는 (상효, 중효, 초효) 양음. True=양, False=음.
GUA_NAMES = {
    (True, True, True):    "건 乾",
    (False, True, True):   "태 兌",
    (True, False, True):   "리 離",
    (False, False, True):  "진 震",
    (True, True, False):   "손 巽",
    (False, True, False):  "감 坎",
    (True, False, False):  "간 艮",
    (False, False, False): "곤 坤",
}


# ── 단계 분류 ──
def classify_stage(s, m, l):
    if s > m > l:
        return 1
    if m > s > l:
        return 2
    if m > l > s:
        return 3
    if l > m > s:
        return 4
    if l > s > m:
        return 5
    if s > l > m:
        return 6
    return 0


def stage_confidence(ema5, ema20, ema40):
    s, m, l = ema5.iloc[-1], ema20.iloc[-1], ema40.iloc[-1]
    vals = sorted([s, m, l], reverse=True)
    if vals[1] == 0 or vals[2] == 0:
        return 0
    gap1 = abs(vals[0] - vals[1]) / vals[1] * 100
    gap2 = abs(vals[1] - vals[2]) / vals[2] * 100
    return int(round(min(100, min(gap1, gap2) / 3.0 * 100)))


def stage_days(ema5, ema20, ema40, max_look=120):
    cur = classify_stage(ema5.iloc[-1], ema20.iloc[-1], ema40.iloc[-1])
    if cur == 0:
        return 0, 0
    days = 1
    last = len(ema5) - 1
    for i in range(last - 1, max(-1, last - max_look), -1):
        if classify_stage(ema5.iloc[i], ema20.iloc[i], ema40.iloc[i]) == cur:
            days += 1
        else:
            break
    return cur, days


# ── 괘상 ──
def _slope_up(series, lookback=3):
    if len(series) <= lookback:
        return None
    return bool(series.iloc[-1] > series.iloc[-1 - lookback])


def gua_raw(ema5, ema20, ema40, close):
    e5 = float(ema5.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e40 = float(ema40.iloc[-1])
    disp_5_20 = abs(e5 - e20) / e20 * 100 if e20 else 0.0
    disp_5_40 = abs(e5 - e40) / e40 * 100 if e40 else 0.0
    last5 = close.iloc[-5:]
    price = float(close.iloc[-1])
    hi5 = bool(price >= last5.max())
    lo5 = bool(price <= last5.min())
    up20 = _slope_up(ema20, GUA_SLOPE_LOOKBACK)
    up40 = _slope_up(ema40, GUA_SLOPE_LOOKBACK)
    return {"e5": e5, "e20": e20, "e40": e40,
            "disp_5_20": disp_5_20, "disp_5_40": disp_5_40,
            "hi5": hi5, "lo5": lo5,
            "up20": (False if up20 is None else up20),
            "up40": (False if up40 is None else up40)}


def _yao5_by_stage(stage, raw, avg_5_20, avg_5_40):
    over_5_20 = raw["disp_5_20"] > avg_5_20
    over_5_40 = raw["disp_5_40"] > avg_5_40
    hi5, lo5 = raw["hi5"], raw["lo5"]
    if stage == 1:
        return not (over_5_20 or lo5)
    if stage == 2:
        return hi5
    if stage == 3:
        return over_5_40 or hi5
    if stage == 4:
        return over_5_20 or hi5
    if stage == 5:
        return hi5
    if stage == 6:
        return not (over_5_20 or lo5)
    return False


def _yao_20_40_by_order(stage, up20, up40):
    if stage == 3:
        return True, up40
    if stage == 4:
        return up20, True
    if stage == 1:
        return up20, up40
    if stage == 6:
        return up20, up40
    if stage == 5:
        return up20, True
    if stage == 2:
        return True, up40
    return up20, up40


def gua_from_raw(stage, raw, avg_5_20, avg_5_40):
    if not raw or stage == 0:
        return {"order": ["5", "20", "50"], "yaos": (None, None, None)}
    yao5 = _yao5_by_stage(stage, raw, avg_5_20, avg_5_40)
    yao20, yao40 = _yao_20_40_by_order(stage, raw["up20"], raw["up40"])
    items = [("5", raw["e5"], yao5), ("20", raw["e20"], yao20),
             ("50", raw["e40"], yao40)]
    items.sort(key=lambda x: x[1], reverse=True)
    return {"order": [it[0] for it in items],
            "yaos": tuple(it[2] for it in items)}


def gua_key(gua):
    if not gua or gua.get("yaos") is None or gua["yaos"][0] is None:
        return "-"
    return "".join("양" if y else "음" for y in gua["yaos"])


def gua_name(yao_tuple):
    return GUA_NAMES.get(tuple(bool(y) for y in yao_tuple), "-")


# ── 보조 지표 ──
def macd_stars(close):
    if len(close) < 35:
        return 0
    macd = (close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean())
    signal = macd.ewm(span=9, adjust=False).mean()
    m, s = macd.iloc[-1], signal.iloc[-1]
    above_signal = m >= s
    above_zero = m >= 0
    if above_signal and above_zero:
        return 3
    if above_signal and not above_zero:
        return 2
    if (not above_signal) and above_zero:
        return 1
    return 0


def macd_rising(close):
    if len(close) < 35:
        return False
    macd = (close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean())
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    if len(hist) < 2:
        return False
    return bool(hist.iloc[-1] > hist.iloc[-2])


def vol_trend(df, recent=5, base=5):
    if "Volume" not in df.columns or len(df) < recent + base:
        return None
    v = df["Volume"].dropna()
    if len(v) < recent + base:
        return None
    r = v.iloc[-recent:].mean()
    b = v.iloc[-(recent + base):-recent].mean()
    if b == 0:
        return None
    return float(r / b)


def calc_rsi(close, period=14):
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    v = rsi.iloc[-1]
    return float(v) if v == v else 50.0


def calc_mfi(df, period=14):
    need = ("High", "Low", "Close", "Volume")
    if not all(c in df.columns for c in need) or len(df) < period + 1:
        return None
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    tp = (high + low + close) / 3
    mf = tp * vol
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, float("nan"))
    mfi = 100 - 100 / (1 + mfr)
    v = mfi.iloc[-1]
    return float(v) if v == v else 50.0


def heikin_ashi(df):
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    ha_open = [(df["Open"].iloc[0] + df["Close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["close"].iloc[i - 1]) / 2)
    ha["open"] = ha_open
    ha["high"] = pd.concat([df["High"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["Low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def ha_wl(df):
    need = ("Open", "High", "Low", "Close")
    if not all(c in df.columns for c in need) or len(df) < 2:
        return "-"
    ha = heikin_ashi(df)
    prev = (ha["open"].iloc[-2] + ha["close"].iloc[-2]) / 2
    cur = (ha["open"].iloc[-1] + ha["close"].iloc[-1]
           + ha["high"].iloc[-1] + ha["low"].iloc[-1]) / 4
    return "W" if (cur - prev) > 0 else "L"


def period_return(close, n):
    if len(close) <= n:
        return None
    return (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100


def stage_flow(close, months=6):
    """이미 받은 종가 시리즈로 최근 N개월 단계 흐름 요약
    (예: '4(30일)→6(12일)→1(45일)'). 전환점만 추출, 마지막=현재 단계."""
    if close is None or len(close) < SPAN_LONG + 5:
        return "-"
    e5 = close.ewm(span=SPAN_SHORT, adjust=False).mean()
    e20 = close.ewm(span=SPAN_MID, adjust=False).mean()
    e40 = close.ewm(span=SPAN_LONG, adjust=False).mean()
    cutoff = close.index[-1] - pd.Timedelta(days=months * 31)
    segs = []
    prev = None
    for d in close.index:
        if d < cutoff:
            continue
        st = classify_stage(e5[d], e20[d], e40[d])
        if st == 0:
            continue
        if st != prev:
            segs.append([st, d, d])
            prev = st
        else:
            segs[-1][2] = d
    if not segs:
        return "-"
    parts = []
    for st, s_date, e_date in segs:
        days = (e_date - s_date).days + 1
        parts.append(f"{st}({days}일)")
    return "→".join(parts)


def prev_flow_text(flow):
    """현재 단계는 제외하고 이전 단계 최근 3개만 (예: '4(30일)-6(12일)')."""
    if not flow or flow == "-" or "→" not in flow:
        return "-"
    prev_parts = flow.split("→")[:-1]
    if not prev_parts:
        return "-"
    return "-".join(prev_parts[-3:])


# ── 한 종목 분석 (DataFrame → 결과 dict, 괘상은 2-pass에서 확정) ──
def analyze_one(code, name, market, mktcap, df):
    """OHLCV DataFrame을 받아 한 종목 분석 결과 dict 반환. 부족하면 None."""
    if df is None or len(df) < SPAN_LONG + 5:
        return None
    close = df["Close"].dropna()
    if len(close) < SPAN_LONG + 5:
        return None
    ema5 = close.ewm(span=SPAN_SHORT, adjust=False).mean()
    ema20 = close.ewm(span=SPAN_MID, adjust=False).mean()
    ema40 = close.ewm(span=SPAN_LONG, adjust=False).mean()
    es, em, el = ema5.iloc[-1], ema20.iloc[-1], ema40.iloc[-1]
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    st = classify_stage(es, em, el)
    return {
        "code": code, "name": name, "market": market,
        "price": price,
        "change": (price - prev) / prev * 100 if prev else 0.0,
        "mktcap": (float(mktcap) if mktcap is not None and pd.notna(mktcap) else None),
        "stage": st,
        "gua": None,
        "gua_raw": gua_raw(ema5, ema20, ema40, close),
        "stars": macd_stars(close),
        "disp20": (price - em) / em * 100 if em else 0.0,
        "disp50": (price - el) / el * 100 if el else 0.0,
        "rsi": calc_rsi(close),
        "mfi": calc_mfi(df),
        "ha": ha_wl(df),
        "days": stage_days(ema5, ema20, ema40)[1],
        "conf": stage_confidence(ema5, ema20, ema40),
        "ma20_up": bool(em > ema20.iloc[-2]) if len(ema20) > 1 else False,
        "ma5_up": bool(ema5.iloc[-1] > ema5.iloc[-1 - GUA_SLOPE_LOOKBACK])
                  if len(ema5) > GUA_SLOPE_LOOKBACK else False,
        "macd_up": macd_rising(close),
        "voltr": vol_trend(df),
        "flow": prev_flow_text(stage_flow(close)),
        "_df": df,   # 차트 데이터 추출용 (JSON 직전에 제거)
    }


def chart_payload(df, days=120):
    """후보 종목 차트용 경량 데이터 (데스크톱 실행파일과 동일 구성):
    캔들(OHLC) + EMA5/20/50 + 거래량 + MACD(라인·시그널·히스토그램)."""
    if df is None or len(df) < 5:
        return None
    close = df["Close"]
    e5 = close.ewm(span=SPAN_SHORT, adjust=False).mean()
    e20 = close.ewm(span=SPAN_MID, adjust=False).mean()
    e40 = close.ewm(span=SPAN_LONG, adjust=False).mean()
    # MACD (12, 26, 9)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    d = df.tail(days)
    n = len(d)

    def tail_round(series):
        return [round(float(x), 2) for x in series.tail(days).tolist()]

    def col(name):
        return [round(float(x), 2) for x in d[name].tolist()]

    has_vol = "Volume" in d.columns
    return {
        "t": [str(x)[:10] for x in d.index],
        "o": col("Open"), "h": col("High"),
        "l": col("Low"), "c": col("Close"),
        "v": [int(x) for x in d["Volume"].fillna(0).tolist()] if has_vol else [0] * n,
        "e5": tail_round(e5), "e20": tail_round(e20), "e40": tail_round(e40),
        "macd": tail_round(macd), "signal": tail_round(signal),
        "hist": tail_round(hist),
    }


def finalize_gua(results):
    """2-pass: 단계별 평균 이격으로 괘상 확정 (in-place)."""
    sums = {}
    for r in results:
        raw = r.get("gua_raw")
        st = r["stage"]
        if not raw or st == 0:
            continue
        acc = sums.setdefault(st, [0.0, 0.0, 0])
        acc[0] += raw["disp_5_20"]
        acc[1] += raw["disp_5_40"]
        acc[2] += 1
    avg = {st: (a[0] / a[2], a[1] / a[2]) for st, a in sums.items() if a[2]}
    for r in results:
        raw = r.get("gua_raw")
        st = r["stage"]
        a5_20, a5_40 = avg.get(st, (raw["disp_5_20"] if raw else 0.0,
                                    raw["disp_5_40"] if raw else 0.0))
        r["gua"] = gua_from_raw(st, raw, a5_20, a5_40)


# ── 관심 후보 선별 ──
def _pick_score(r):
    st = r["stage"]
    rsi = r.get("rsi") or 50
    mfi = r.get("mfi") or 50
    stars = r.get("stars", 0)
    vt = r.get("voltr") or 1.0
    g5 = r.get("gap_5_20", 0)
    g25 = r.get("gap_20_50", 0)
    if st == 1:
        return (stars, -vt, rsi + mfi)
    if st in (5, 6):
        return (stars, -vt, rsi + mfi)
    if st == 2:
        return (0 if (r.get("macd_up") or r.get("ma20_up")) else 1, vt)
    return (-(g5 + g25),)


def filter_pick(results):
    """관심 후보: 괘상 '건(양양양)' 단계별 할당 선별.
    단계 할당 = round(50 × (단계 종목수/전체) × 0.5), 나머지는 1·5·6·2 순 충원."""
    max_total = PICK_MAX_TOTAL
    fill_order = (1, 5, 6, 2, 3, 4)
    total = sum(1 for r in results if r.get("stage", 0) != 0)
    stage_total = {k: 0 for k in range(1, 7)}
    for r in results:
        st = r.get("stage", 0)
        if st in stage_total:
            stage_total[st] += 1
    GEON = "양양양"
    geon_by_stage = {k: [] for k in range(1, 7)}
    for r in results:
        st = r.get("stage", 0)
        if st in geon_by_stage and gua_key(r.get("gua")) == GEON:
            geon_by_stage[st].append(r)
    for st in geon_by_stage:
        geon_by_stage[st].sort(key=_pick_score)
    quota = {k: 0 for k in range(1, 7)}
    if total > 0:
        for k in range(1, 7):
            quota[k] = int(round(max_total * (stage_total[k] / total) * 0.5))
    remainder = max_total - sum(quota.values())
    i = 0
    while remainder > 0:
        quota[fill_order[i % len(fill_order)]] += 1
        remainder -= 1
        i += 1
    picked = []
    used = set()
    carry = 0
    for st in fill_order:
        want = quota.get(st, 0) + carry
        avail = [r for r in geon_by_stage.get(st, []) if id(r) not in used]
        take = avail[:want]
        for r in take:
            used.add(id(r))
        picked.extend(take)
        carry = want - len(take)
    if len(picked) < max_total:
        for st in fill_order:
            if len(picked) >= max_total:
                break
            for r in geon_by_stage.get(st, []):
                if len(picked) >= max_total:
                    break
                if id(r) not in used:
                    used.add(id(r))
                    picked.append(r)
    grp = {1: 0, 6: 1, 2: 2, 5: 3, 3: 4, 4: 5}
    picked.sort(key=lambda x: (grp.get(x["stage"], 9), _pick_score(x)))
    return picked[:max_total]


def filter_riri(results):
    """4단계 '리(離)' + MACD 상승추세 + 5일선 상승추세, 최대 10개."""
    riri_shape = next((k for k, v in GUA_NAMES.items()
                       if v.startswith("리")), (True, False, True))
    riri_key = "".join("양" if y else "음" for y in riri_shape)
    cand = []
    for r in results:
        if r.get("stage") != 4:
            continue
        if gua_key(r.get("gua")) != riri_key:
            continue
        if not r.get("macd_up", False):
            continue
        if not r.get("ma5_up", False):
            continue
        cand.append(r)
    cand.sort(key=lambda x: (-(x.get("voltr") or 1.0), -(x.get("conf") or 0)))
    return cand[:RIRI_MAX]
