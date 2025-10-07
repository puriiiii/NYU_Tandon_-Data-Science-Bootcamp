#2. Iterate through the following list of animals and print each one in all caps.

#  animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']

animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']
for animal in animals: 
	a_list = list(animal)
	for i in range(len(a_list)): 
		a_list[i] = a_list[i].upper()
	animal_word = ''.join(a_list)
	print(animal_word)
		