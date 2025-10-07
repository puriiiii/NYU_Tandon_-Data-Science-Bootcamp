#. Write a program that iterates from 1 to 20, printing each number and whether it's odd or even.

i = 1 
helper = ["Even","Odd"]
while (i<21): 
	print(f"{i} is {helper[i%2]}")
	i+=1
