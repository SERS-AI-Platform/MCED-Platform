---
applyTo: "**/*.sql"
---

# SQL 작성 규칙

## 1. 사용자 컨텍스트
사용자는 SQL 초보 → expert로 성장 중. **feedback 기반 학습** 선호.

- 정답만 주지 말고 **왜 이렇게 쓰는지** 주석/설명 포함
- 초보가 흔히 하는 실수·안티패턴 발견 시 즉시 지적 (단순 수정 X, 이유 설명)
- 점진적 레벨 향상: 기본 SELECT → JOIN → GROUP BY → Window → CTE → 성능 튜닝
- 자동 실행하지 말고 `.sql` 파일로 전달 → DBeaver에서 직접 실행하도록
- 결과 해석도 본인이 먼저 해볼 수 있도록 "예상 결과" 힌트는 주되 정답은 늦게 공개
- 작성한 쿼리에 대한 리뷰/피드백 요청 시 성능·가독성·안전성 관점에서 비평

## 2. 곡예 금지

실무에서 실제로 쓰는 수준 넘지 말 것:

❌ **금지**:
- `xpath`, `query_to_xml`, `LATERAL` + 동적 SQL 곡예
- "이게 더 고급이니까" 라며 복잡도 올리기
- "3가지 방법" 식으로 난이도 계단 만들기

✅ **권장**:
- 표준 `SELECT / JOIN / GROUP BY / CTE / Window` 수준
- 메타정보 조회는 `information_schema` 기본 + DBeaver GUI로 충분
- 현실적인 1~2가지 풀이만

복잡한 쿼리가 정말 필요하면 "**왜 이게 표준보다 복잡해지는지**" 명시.

## 3. 쿼리 마다 주석 필수
- 헤더 주석: 목적, 입력 가정, 예상 결과
- 인라인 주석: WHY 중심 (WHAT은 코드가 이미 말함)

```sql
-- 목적: 7대 암종별 평균 AUC 계산 (2026 Q1 실험)
-- 가정: experiments 테이블에 valid_run = TRUE만 사용
-- 예상: 7행, AUC 0.8~0.99 범위
WITH valid_runs AS (
    SELECT *
    FROM experiments
    WHERE valid_run = TRUE  -- 망가진 run 제외
)
SELECT
    cancer_type,
    AVG(auc) AS mean_auc
FROM valid_runs
GROUP BY cancer_type
ORDER BY mean_auc DESC;
```
