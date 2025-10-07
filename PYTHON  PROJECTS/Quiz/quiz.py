"""
Quiz Game 🎯

A command-line Python quiz game that asks random questions to the user, 
checks their answers, keeps track of the score, and provides feedback 
based on performance. The user can choose how many questions to play 
and can replay the quiz multiple times.

Features:
- Randomly selects questions from a predefined question bank
- Validates user input for number of questions
- Tracks score and calculates percentage
- Gives feedback based on performance
- Option to play multiple rounds

"""

import random 

questions={
    "What is the capital of India?":"new delhi",
    "What is the chemical symbol for water?":"h2o",
    "Who wrote the play 'Romeo and Juliet'?":"shakespeare",
    "What is the square root of 64?":"8",
    "Which planet is known as the Red Planet?":"mars",
    "What is the largest mammal in the world?":"blue whale",
    "How many continents are there on Earth?":"7",
    "Who is the current CEO of Tesla? (as of 2025)":"elon musk",
    "Which language is used to build websites? HTML, C, or Python?":"html",
    "What is 15 divided by 3?":"5"
}

user_name=input("Hey user,\nplease enter your name: ")

def game_quiz():
    while True:
        try:
           quize_times=int(input(f"how many times do you want to play quiz , {user_name} "))
           if(quize_times>len(questions)):
            print(f"Only {len(questions)} questions are available. Adjusting the quiz length.")
            quize_times=len(questions)
            break 
           elif(quize_times<=0):
            print("Please enter a number greater than 0.")
            continue 
           else:
               break
        except ValueError:
            print("Invalid input.\nPlease enter a valid number.")
            continue 
    
    score=0
    picked_questions=random.sample(list(questions.keys()),quize_times)
    for i in picked_questions:
        answers_user=input(f"{i}\nEnter your answer please: ").lower().strip()
        if answers_user==questions[i].lower().strip():
            score+=1
            print("Correct answer!!")
        else:
            score+=0
            print(f"that's an incorrect answer!!\ncorrect answer is {questions[i]}")
    
    print(f"You got {score}/{quize_times}")

    percentage=(score/quize_times)*100

    if(score==quize_times):
       print(f"Excellent!! {user_name}")
    elif(percentage>=80):
       print(f"You did great!! {user_name}")
    elif(percentage>=50):
       print(f"Nice!! {user_name}")
    else:
       print(f"You neeed to improve,{user_name}")
    
    repeat=input("Do you want to play again?(yes or no) ")
        
    if(repeat.lower().strip()=="yes"):
            game_quiz()
    else:
        print("Thank you for playing!!!")
        quit()

game_quiz()