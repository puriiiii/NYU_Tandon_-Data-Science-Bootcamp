#Write a program that takes a word as an input and print the number of vowels in the word.

input_word = input("Enter a word : ")
vowels = "aeiou"
vowel_count = 0
for letter in input_word: 
	if letter.lower() in vowels: 
		vowel_count += 1	

print(f"Number of Vowels in {input_word} is {vowel_count}")