# Advanced SQL Interview Questions and Solutions

## 1. Window Functions

### Question: Calculate Running Total
Given a sales table with columns (date, amount), write a query to calculate the running total of sales.

```sql
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM sales;
```

### Question: Rank Employees by Salary Within Department
Write a query to rank employees by salary within each department.

```sql
SELECT 
    department,
    employee_name,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
FROM employees;
```

## 2. Complex Joins

### Question: Self Join for Employee Hierarchy
Write a query to display employee and their manager names.

```sql
SELECT 
    e.employee_name as employee,
    m.employee_name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;
```

### Question: Multiple Joins with Aggregation
Find total sales by product category and region.

```sql
SELECT 
    c.category_name,
    r.region_name,
    SUM(s.amount) as total_sales
FROM sales s
JOIN products p ON s.product_id = p.product_id
JOIN categories c ON p.category_id = c.category_id
JOIN regions r ON s.region_id = r.region_id
GROUP BY c.category_name, r.region_name;
```

## 3. Subqueries

### Question: Employees Earning More Than Average
Find employees who earn more than the average salary.

```sql
SELECT employee_name, salary
FROM employees
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
);
```

### Question: Products with Above Average Sales
Find products that have above-average sales in their category.

```sql
SELECT 
    p.product_name,
    c.category_name,
    s.total_sales
FROM products p
JOIN (
    SELECT 
        product_id,
        SUM(amount) as total_sales
    FROM sales
    GROUP BY product_id
) s ON p.product_id = s.product_id
JOIN categories c ON p.category_id = c.category_id
WHERE s.total_sales > (
    SELECT AVG(total_sales)
    FROM (
        SELECT product_id, SUM(amount) as total_sales
        FROM sales
        GROUP BY product_id
    ) avg_sales
);
```

## 4. CTEs (Common Table Expressions)

### Question: Find Repeat Customers
Write a query to find customers who have made more than one purchase.

```sql
WITH customer_purchases AS (
    SELECT 
        customer_id,
        COUNT(*) as purchase_count
    FROM orders
    GROUP BY customer_id
)
SELECT 
    c.customer_name,
    cp.purchase_count
FROM customers c
JOIN customer_purchases cp ON c.customer_id = cp.customer_id
WHERE cp.purchase_count > 1;
```

## 5. Advanced Grouping

### Question: Sales Report with Rollup
Create a sales report with subtotals for each level of hierarchy.

```sql
SELECT 
    COALESCE(region, 'Total') as region,
    COALESCE(product_category, 'Subtotal') as category,
    SUM(amount) as total_sales
FROM sales s
JOIN regions r ON s.region_id = r.region_id
JOIN products p ON s.product_id = p.product_id
GROUP BY ROLLUP(region, product_category);
```

## 6. Date and Time Functions

### Question: Monthly Revenue Growth
Calculate month-over-month revenue growth.

```sql
WITH monthly_revenue AS (
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(amount) as revenue
    FROM sales
    GROUP BY DATE_TRUNC('month', date)
)
SELECT 
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
    (revenue - LAG(revenue) OVER (ORDER BY month)) / 
        LAG(revenue) OVER (ORDER BY month) * 100 as growth_percent
FROM monthly_revenue;
```

## 7. Performance Optimization

### Question: Explain how you would optimize this query:
```sql
SELECT 
    c.customer_name,
    COUNT(o.order_id) as total_orders,
    SUM(o.amount) as total_amount
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
GROUP BY c.customer_id, c.customer_name
HAVING total_amount > 1000;
```

Answer:
1. Add indexes:
   ```sql
   CREATE INDEX idx_order_date ON orders(order_date);
   CREATE INDEX idx_customer_id ON orders(customer_id);
   ```
2. Materialize the aggregates if frequently queried:
   ```sql
   CREATE MATERIALIZED VIEW customer_order_stats AS
   SELECT 
       customer_id,
       COUNT(order_id) as total_orders,
       SUM(amount) as total_amount
   FROM orders
   WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)
   GROUP BY customer_id;
   ```
3. Partition large tables by date range if applicable.

## 8. Error Handling

### Question: Implementing Safe Division
Write a query to handle division by zero errors.

```sql
SELECT 
    department,
    total_salary,
    total_employees,
    CASE 
        WHEN total_employees = 0 THEN 0
        ELSE total_salary / total_employees 
    END as avg_salary
FROM department_stats;
```

## 9. Data Quality Checks

### Question: Write queries to validate data quality

```sql
-- Check for duplicate records
SELECT 
    column1, column2, COUNT(*)
FROM table_name
GROUP BY column1, column2
HAVING COUNT(*) > 1;

-- Check for orphaned foreign keys
SELECT DISTINCT a.foreign_key_id
FROM table_a a
LEFT JOIN table_b b ON a.foreign_key_id = b.id
WHERE b.id IS NULL;

-- Validate date ranges
SELECT *
FROM orders
WHERE order_date > delivery_date
   OR order_date > CURRENT_DATE;
```
