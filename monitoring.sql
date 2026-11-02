-- Portfolio monitoring query run by run_pipeline.py on the scored table (SQLite)
SELECT score_band,
       COUNT(*)                         AS accounts,
       ROUND(AVG(is_default) * 100, 2)     AS observed_default_pct,
       ROUND(AVG(pd_hat) * 100, 2)      AS predicted_pd_pct
FROM scored
WHERE sample = 'out_of_time'
GROUP BY score_band
ORDER BY score_band;
