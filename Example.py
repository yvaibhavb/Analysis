#Python classes 
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        print("Some generic animal sound")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, species="Dog")
        self.breed = breed
    
    def make_sound(self):
        print("Woof!")

dog1 = Dog("Fido", "Labrador")
#print(dog1.name)
#print(dog1.species)

#Python Array
myAnimal = [Dog("F1", "v1"), Dog("F2", "V2")]
print(myAnimal[0].breed)

for i in myAnimal:
    print(i.breed, i.name);

#print all object in array
testArry = [1, 2, 3.14, [1,2,3]]
print(*testArry)

#Dictonary 
myDict = {"d1":1, "d2":2}
print(myDict["d1"])
for key, value in myDict:
    print(f"{key}, {value}")