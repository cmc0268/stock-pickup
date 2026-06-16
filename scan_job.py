# -*- coding: utf-8 -*-
"""
관심 후보 스캔 배치 (서버/GitHub Actions 전용)
- KOSPI+KOSDAQ 시총 상위 종목을 병렬로 내려받아 분석
- 괘상 '건' 관심 후보 + 4단계 '리' 별도 선별
- 결과를 data/picks.json 으로 저장 (웹앱이 이 파일만 읽음)

실행: python scan_job.py
환경변수(선택):
  TOP_KOSPI (기본 380), TOP_KOSDAQ (기본 420), MAX_WORKERS (기본 24)
"""
import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import FinanceDataReader as fdr

import scanner_core as core

KST = timezone(timedelta(hours=9))
TOP_KOSPI = int(os.environ.get("TOP_KOSPI", "380"))
TOP_KOSDAQ = int(os.environ.get("TOP_KOSDAQ", "420"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "24"))
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "picks.json")


def get_stock_list():
    """KOSPI+KOSDAQ 종목 목록 (스팩 제외), 시총 상위로 한정."""
    frames = []
    for market, label, top in (("KOSPI", "피", TOP_KOSPI),
                               ("KOSDAQ", "닥", TOP_KOSDAQ)):
        try:
            df = fdr.StockListing(market)
        except Exception as e:
            print(f"[목록 실패] {market}: {e}", file=sys.stderr)
            continue
        code_col = next((c for c in ("Code", "Symbol", "단축코드", "종목코드")
                         if c in df.columns), None)
        name_col = next((c for c in ("Name", "Korean Name", "한글종목명", "종목명")
                         if c in df.columns), None)
        cap_col = next((c for c in ("Marcap", "MarketCap", "Market Cap", "시가총액")
                        if c in df.columns), None)
        if code_col is None or name_col is None:
            continue
        sub = pd.DataFrame({
            "code": df[code_col].astype(str).str.zfill(6),
            "name": df[name_col].astype(str),
            "mktcap": df[cap_col] if cap_col else None,
            "market": label,
        })
        sub = sub[~sub["name"].str.contains("스팩", na=False)]
        sub = sub.dropna(subset=["code", "name"]).drop_duplicates("code")
        if cap_col and sub["mktcap"].notna().any():
            sub = sub.sort_values("mktcap", ascending=False).head(top)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["code", "name", "mktcap", "market"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("code")


def fetch_and_analyze(row, start):
    code = row["code"]
    try:
        df = fdr.DataReader(code, start)
    except Exception:
        return None
    return core.analyze_one(code, row["name"], row["market"],
                            row.get("mktcap"), df)


def trim(r):
    """JSON 출력용 슬림화 (gua_raw 등 내부 필드 제거, 표시값만)."""
    gua = r.get("gua") or {}
    yaos = gua.get("yaos")
    shape = ("".join("양" if y else "음" for y in yaos)
             if yaos and yaos[0] is not None else "-")
    # 괘상 명칭 (상,중,초 = yaos 순서)
    name = core.gua_name(yaos) if yaos and yaos[0] is not None else "-"
    return {
        "code": r["code"],
        "name": r["name"],
        "market": "코스피" if r.get("market") == "피" else
                  "코스닥" if r.get("market") == "닥" else r.get("market", ""),
        "stage": r["stage"],
        "stage_label": core.STAGE_LABEL.get(r["stage"], ""),
        "stage_color": core.STAGE_COLOR.get(r["stage"], "#888"),
        "gua_shape": shape,         # 상효→초효 양음 (예: 양양양)
        "gua_name": name,           # 예: 건 乾
        "gua_order": gua.get("order", []),  # 효 위치 이평선 (예: ["5","20","50"])
        "price": round(r["price"], 2),
        "change": round(r.get("change") or 0, 2),
        "disp20": round(r.get("disp20") or 0, 2),
        "disp50": round(r.get("disp50") or 0, 2),
        "rsi": round(r["rsi"], 1) if r.get("rsi") is not None else None,
        "mfi": round(r["mfi"], 1) if r.get("mfi") is not None else None,
        "ha": r.get("ha", "-"),
        "flow": r.get("flow", "-"),
        "stars": r.get("stars", 0),
        "days": r.get("days", 0),
        "conf": r.get("conf", 0),
        "mktcap": r.get("mktcap"),
        "chart": core.chart_payload(r.get("_df")),
    }


def main():
    t0 = time.time()
    stocks = get_stock_list()
    n = len(stocks)
    if n == 0:
        # 종목 목록 실패 시: 기존 결과를 절대 덮어쓰지 않음 (자료 보존)
        print("종목 목록을 받지 못했습니다. 기존 picks.json 보존.", file=sys.stderr)
        sys.exit(1)
    start = (datetime.now(KST) - timedelta(days=core.LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"[스캔] {n}종목, 워커 {MAX_WORKERS}개, 시작일 {start}")

    results = []
    done = 0
    rows = [dict(r._asdict()) if hasattr(r, "_asdict") else r
            for r in stocks.to_dict("records")]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_and_analyze, row, start): row for row in rows}
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception:
                res = None
            if res is not None:
                results.append(res)
            if done % 100 == 0:
                print(f"  진행 {done}/{n} (수집 {len(results)})")

    core.finalize_gua(results)
    picks = core.filter_pick(results)
    riri = core.filter_riri(results)

    # 단계별 요약 (전체 분석 종목 기준 상승/하락)
    by_stage = {}
    for r in results:
        by_stage.setdefault(r["stage"], []).append(r)
    stage_summary = []
    for st in (1, 6, 2, 5, 3, 4):
        lst = by_stage.get(st, [])
        up = sum(1 for x in lst if (x.get("change") or 0) >= 0)
        stage_summary.append({
            "stage": st, "label": core.STAGE_LABEL[st],
            "color": core.STAGE_COLOR[st],
            "count": len(lst), "up": up, "down": len(lst) - up,
        })

    out = {
        "updated_at": datetime.now(KST).isoformat(),
        "scanned": len(results),
        "universe": n,
        "elapsed_sec": round(time.time() - t0, 1),
        "stage_summary": stage_summary,
        "picks": [trim(r) for r in picks],
        "riri": [trim(r) for r in riri],
    }
    # 분석 결과가 비정상적으로 적으면(데이터 대량 실패) 기존 자료 보존
    if len(results) < max(30, int(n * 0.3)):
        print(f"분석 {len(results)}/{n} — 너무 적어 실패로 간주. "
              f"기존 picks.json 보존(덮어쓰지 않음).", file=sys.stderr)
        sys.exit(1)
    _write(out)
    print(f"[완료] 분석 {len(results)}/{n} · 관심후보 {len(picks)} · "
          f"리 {len(riri)} · {out['elapsed_sec']}s → {OUT_PATH}")


def _write(obj):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
