Create Table employees (
	id serial Primary key,
	
	name VARCHAR(100) NOT NULL,

    department VARCHAR(100),

    salary DECIMAL(10,2)
);
Select * From employees;

Insert Into employees (
	name,
	department,
	salary
) 
Values 
(
'Rahul',
'Backend',
68000
),

(
'Priya',
'HR',
55000
),

(
'Aman',
'AI',
82000
),

(
'Neha',
'Finance',
60000
)

Select * From employees Order By salary

update employees 
set salary = 72000 
where id = 2;

ALTER TABLE employees
ADD COLUMN email VARCHAR(100);

SELECT
    name AS employee_name,
    salary AS monthly_salary
FROM employees;
