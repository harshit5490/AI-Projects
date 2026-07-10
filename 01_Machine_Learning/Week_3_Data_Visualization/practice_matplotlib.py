import matplotlib.pyplot as plt
# line Chart

# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
# plt.figure(figsize=(8,5))
# plt.grid(True)
# plt.plot(x,y, color = "red",linewidth = 3,marker = "o")
# plt.title("My First Graph")
# plt.xlabel("X Values")
# plt.ylabel("Y Values")
# plt.show()

# Bar Chart 

departments = ["AI", "IT", "HR", "Sales"]
employees = [40, 30, 15, 20]
colors = [
    "red",
    "green",
    "blue",
    "orange"
]
# plt.bar(departments, employees,color =colors,width=0.5,edgecolor = "black")
# plt.barh(departments, employees,color =colors,edgecolor = "black")
# bars = plt.bar(departments,employees)
# for bar in bars:
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         height,
#         str(height),
#         ha="center",
#         va="bottom"
#     )


# plt.title("Employees in Each Department")
# plt.xlabel("Department")
# plt.ylabel("Number of Employees")
# plt.show()

#Sactter Chart

# experience = [1,2,3,4,5,6]

# salary = [30000,40000,45000,60000,70000,85000]

# plt.scatter(experience, salary,s=100,alpha=0.6)

# plt.title("Experience vs Salary")

# plt.xlabel("Experience (Years)")

# plt.ylabel("Salary")

# plt.grid(True)

# plt.show()


#Histogram Chart

# marks = [45,56,72,68,90,85,76,65,74,81,50,48,66,71,59,63,77,88,91,53]

# plt.hist(marks,bins=5,edgecolor="black")

# plt.title("Distribution of Student Marks")
# plt.xlabel("Marks")
# plt.ylabel("Number of Students")

# plt.show()

# Boxplot Chart

# salary = [
# 25000,28000,30000,32000,
# 35000,37000,40000,42000,
# 45000,48000,50000,52000,
# 55000,60000,65000,70000,
# 75000,80000,90000,500000
# ] 
# plt.boxplot(salary,patch_artist=True,
#     boxprops=dict(facecolor="lightblue"),showmeans=True) # add vert = false for horizontal chart
# plt.title("Employee Salary Distribution")

# plt.ylabel("Salary")

# plt.show()

# Pie Chart

# departments = ["AI", "IT", "HR", "Sales"]

# employees = [40, 30, 20, 10]
# explode = [0.1, 0, 0, 0]

# plt.pie(employees,labels=departments,autopct="%1.1f%%",explode=explode,startangle=90,shadow=True)

# plt.show()

# Subplot for multiple graph in one go
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.plot([1,2,3],[2,4,6])
plt.title("Line")

plt.subplot(2,2,2)
plt.bar(["A","B","C"],[5,6,4])
plt.title("Bar")

plt.subplot(2,2,3)
plt.scatter([1,2,3],[5,6,7])
plt.title("Scatter")

plt.subplot(2,2,4)
plt.hist([10,20,30,20,25,35,40])
plt.title("Histogram")

plt.tight_layout() # this is used for no overlapping between the graph

plt.show()