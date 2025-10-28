-- Find users who signed up in Jan 2023 and purchased in BOTH Feb and Mar 2023
SELECT COUNT(DISTINCT u.user_id) AS retained_users
FROM users u
WHERE u.signup_date >= '2023-01-01' 
  AND u.signup_date < '2023-02-01'
  AND EXISTS (
      SELECT 1 
      FROM purchases p 
      WHERE p.user_id = u.user_id 
        AND p.purchase_date >= '2023-02-01' 
        AND p.purchase_date < '2023-03-01'
  )
  AND EXISTS (
      SELECT 1 
      FROM purchases p 
      WHERE p.user_id = u.user_id 
        AND p.purchase_date >= '2023-03-01' 
        AND p.purchase_date < '2023-04-01'
  );