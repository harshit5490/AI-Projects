CREATE TABLE departments (

    department_id SERIAL PRIMARY KEY,

    department_name VARCHAR(50),

    location VARCHAR(50)

);
CREATE TABLE employees (

    employee_id SERIAL PRIMARY KEY,

    employee_name VARCHAR(100),

    department_id INT,

    salary DECIMAL(10,2),

    FOREIGN KEY (department_id)
    REFERENCES departments(department_id)

);
INSERT INTO departments
(department_name, location)

VALUES

('AI','Delhi'),
('Backend','Noida'),
('HR','Mumbai'),
('Finance','Pune');
INSERT INTO employees
(employee_name, department_id, salary)

VALUES

('Harshit',1,85000),

('Rahul',2,70000),

('Aman',1,95000),

('Priya',3,55000),

('Neha',NULL,60000);
SELECT

e.employee_name,

d.department_name,

d.location,

e.salary

FROM employees e

INNER JOIN departments d

ON e.department_id=d.department_id;
SELECT

e.employee_name,

d.department_name

FROM employees e

LEFT JOIN departments d

ON e.department_id=d.department_id;
SELECT

e.employee_name,

d.department_name

FROM employees e

RIGHT JOIN departments d

ON e.department_id=d.department_id;
SELECT

e.employee_name,

d.department_name

FROM employees e

FULL OUTER JOIN departments d

ON e.department_id=d.department_id;
SELECT

e.employee_name,

d.department_name

FROM employees e

CROSS JOIN departments d;
-- SELECT

-- e.name AS employee,

-- m.name AS manager

-- FROM employees e

-- LEFT JOIN employees m

-- ON e.manager_id=m.employee_id;