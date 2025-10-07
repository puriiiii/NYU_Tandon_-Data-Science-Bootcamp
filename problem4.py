#4. Write a program to check if a string is a palindrome or not.

word = input("Enter a word : ").lower()
ptr_1 = 0 
ptr_2 = -1 

for i in range(len(word)//2): 
	if (word[ptr_1] == word[ptr_2]): 
		pallindrome = True
		ptr_1 += 1
		ptr_2 -= 1
	else: 
		pallindrome = False
		break 
	
dic = {True: "pallindrome", 
	   False:"not pallindrome" }
	   
print(f"{word} is {dic[pallindrome]}")

