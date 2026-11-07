-- analysis.sql : starter queries. Paste them into a Snowflake worksheet (USE DATABASE POKER_BOT first).
-- Tables: MATCHES (one row per match) and DECISIONS (one row per decision of YOUR bot).
-- Remember: hand results are noisy (about 25 bb standard deviation per deal). Only trust big samples.

-- 1) Strength per bot version and opponent. Use runs with EPS = 0 (no random exploration).
SELECT BOT_VERSION, OPPONENT, COUNT(*) AS MATCHES,
       ROUND(AVG(MEAN_BB_PER_DEAL), 3) AS AVG_BB_PER_DEAL,
       SUM(IFF(VERDICT = 'WIN', 1, 0)) AS WINS, SUM(IFF(VERDICT = 'CHOP', 1, 0)) AS CHOPS,
       SUM(IFF(VERDICT = 'LOSS', 1, 0)) AS LOSSES,
       ROUND(MAX(MY_COMPUTE_SECONDS_PER_HALF), 2) AS MAX_COMPUTE_S_PER_HALF
FROM MATCHES WHERE EPS = 0
GROUP BY 1, 2 ORDER BY 1, 2;

-- 2) FAIR action comparison. Only EXPLORED = 1 rows have a randomly chosen action, so differences
--    between actions are not biased by what the bot normally prefers.
--    AVG_NET = average chips the whole hand ended up with after taking this action here.
SELECT STREET, FLOOR(EQUITY * 10) / 10 AS EQUITY_BUCKET, ACTION,
       COUNT(*) AS N, ROUND(AVG(HAND_NET_CHIPS), 2) AS AVG_NET
FROM DECISIONS
WHERE EXPLORED = 1 AND EQUITY IS NOT NULL
GROUP BY 1, 2, 3 HAVING COUNT(*) >= 30
ORDER BY STREET, EQUITY_BUCKET, ACTION;

-- 3) Possible bad folds: we folded although our estimated win chance beat the raw pot odds.
--    (Equity here is vs a RANDOM hand, so it overstates our chances against a betting opponent.)
SELECT STREET, COUNT(*) AS FOLDS,
       ROUND(AVG(TO_CALL / (POT + TO_CALL)), 3) AS AVG_PRICE_TO_CALL,
       ROUND(AVG(EQUITY), 3) AS AVG_EQUITY,
       ROUND(AVG(HAND_NET_CHIPS), 2) AS AVG_NET
FROM DECISIONS
WHERE ACTION = 'FOLD' AND EXPLORED = 0 AND EQUITY > TO_CALL / (POT + TO_CALL)
GROUP BY STREET ORDER BY STREET;

-- 4) Position: how do we do on the button vs off it, by street (normal play only)?
SELECT IS_BUTTON, STREET, COUNT(*) AS DECISIONS, ROUND(AVG(HAND_NET_CHIPS), 2) AS AVG_NET
FROM DECISIONS WHERE EXPLORED = 0
GROUP BY 1, 2 ORDER BY 1, 2;

-- 5) Preflop calibration: does our preflop equity estimate predict results? (one row per hand)
SELECT FLOOR(EQUITY * 20) / 20 AS PREFLOP_EQUITY_BUCKET, COUNT(*) AS HANDS,
       ROUND(AVG(HAND_NET_CHIPS), 2) AS AVG_NET, ROUND(AVG(WENT_TO_SHOWDOWN), 2) AS SHOWDOWN_RATE
FROM DECISIONS
WHERE STEP = 1 AND STREET = 'PREFLOP' AND EXPLORED = 0
GROUP BY 1 ORDER BY 1;

-- 6) How does each action choice do against each opponent (normal play)?
SELECT OPPONENT, STREET, ACTION, COUNT(*) AS N, ROUND(AVG(HAND_NET_CHIPS), 2) AS AVG_NET
FROM DECISIONS WHERE EXPLORED = 0
GROUP BY 1, 2, 3 HAVING COUNT(*) >= 50 ORDER BY 1, 2, 3;
