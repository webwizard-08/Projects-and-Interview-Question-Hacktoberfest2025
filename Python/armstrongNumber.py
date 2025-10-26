"""This program checks if a number is armstrong number or not.
Armstrong Number -> Armstrong number is a number that is equal to the sum of cube
of it's digits.
Example -> 153 = 1^3 + 5^3 + 3^3"""

num = int(input("Enter a number: "))
temp = num
result = 0 #This variable will store the cube of digits
while temp > 0:
    digit = temp%10 #Extracts the last digit
    result += (digit**3)
    temp //= 10 #Get rid of the extracted digit

if result == num:
    print("Armstrong Number")
else:
    print("Not Armstrong Number")
