select count(*) from employees;

SELECT COUNT(*)
FROM employees
WHERE city='Delhi';

SELECT SUM(salary)
FROM employees
WHERE department='AI';

SELECT AVG(salary)
FROM employees
WHERE department='AI';

SELECT MIN(joining_date)
FROM employees;

SELECT MAX(experience)
FROM employees;

SELECT

COUNT(*) AS total_employees,

AVG(salary) AS average_salary,

MIN(salary) AS minimum_salary,

MAX(salary) AS highest_salary,

SUM(salary) AS total_salary

FROM employees;

select department, AVG(salary)
from employees 
group by department;

SELECT

department,

AVG(salary) AS avg_salary

FROM employees

GROUP BY department

ORDER BY avg_salary DESC;

SELECT

department,

AVG(salary)

FROM employees

GROUP BY department

HAVING AVG(salary)>80000;