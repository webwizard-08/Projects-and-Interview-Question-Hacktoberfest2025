/*
Project: Hangman Game
Category: Mini Console Project
*/

#include<bits/stdc++.h>
using namespace std;

vector<string> HANGMAN={
    "\n       |\n       |\n       |\n     =====",
    "\n   O   |\n       |\n       |\n     =====",
    "\n   O   |\n   |   |\n       |\n     =====",
    "\n   O   |\n  /|   |\n       |\n     =====",
    "\n   O   |\n  /|\\  |\n       |\n     =====",
    "\n   O   |\n  /|\\  |\n  /    |\n     =====",
    "\n   O   |\n  /|\\  |\n  / \\  |\n     =====",
};

//choose a random secret word
string chooseWord(vector<string> &words){
    if(words.empty()) return "mystery";
    int ind=rand()%(words.size());
    return words[ind];
}

//display secret word with blanks
string displayWord(string &secret, unordered_set<char>& correct){
    string res="";
    for(auto letter: secret){
        if(correct.count(letter)){
            res+=letter;
        }
        else{
            res+="_";
        }
        res+=" ";
    }
    return res;
}

//verify the user guess
bool verifyLetter(char guess, string secret, unordered_set<char> &correct, unordered_set<char> &wrong){
    bool isPresent=false;
    for(auto ch: secret){
        if(ch==guess){
            isPresent=true;
            break;
        }
    }

    if(isPresent){
        correct.insert(guess);
        cout<<"✅ Good Guess! '"<<guess<<"' is in the word."<<endl;
        return true;
    }
    else{
        wrong.insert(guess);
        cout<<"❌ Oops! '"<<guess<<"' is not in the word."<<endl;
        return false;
    }    
}

int main(){
    srand(time(0));
    char playAgain;

    do{
        vector<string> words={"assassin","zombie", "serendipity","happy","quirk","palate", "mystery", "adventure"};
        string secret=chooseWord(words);
        
        unordered_set<char> correct, wrong;
        int lives=7;
        bool won=false;

        cout<<"\n🎯 Welcome to Hangman! 🎯"<<endl;
        cout<<"Guess the secret word!"<<endl;
        cout<<"--------------------------------------------\n"<<endl;
        cout<<"Hint: The secret has "<<secret.size()<<" letters.";
        
        while(lives>0){
            cout<<"\n"<<HANGMAN[7-lives]<<endl;
            cout<<"\nWord: "<<displayWord(secret, correct)<<endl;
            
            cout<<"Wrong guesses: ";
            for(auto ch : wrong){
                cout<<ch<<" ";
            }
            cout<<"\nLives left: "<<lives<<endl;

            string input;
            cout<<"Enter a letter: ";
            cin>>input;
            if(input.size()!=1){
                cout<<"Please enter only a single letter!"<<endl;
                continue;
            }
            char guess=tolower(input[0]);

            if(!isalpha(guess)){
                cout<<"Please enter a valid letter!"<<endl;
                continue;
            }

            if(correct.count(guess) || wrong.count(guess)){
                cout<<"You already guessed that!"<<endl;
                continue;
            }

            if(!verifyLetter(guess, secret, correct, wrong)){
                lives--;
            }

            bool allFound=true;
            for(auto ch: secret){
                if(!correct.count(ch)){
                    allFound=false;
                    break;
                }
            }

            if(allFound){
                won=true;
                break;
            }
        }

        cout<<"\n--------------------------------------\n";
        if(won){
            cout<<"\n🎉 You won! The secret was: "<<secret<<endl;
        }
        else{
            cout<<"\n💀 You lost! The secret was: "<<secret<<endl;
        }
        cout<<"\n--------------------------------------\n";

        cout<<"\nPlay again? (y/n): ";
        cin>>playAgain;

    }while(playAgain=='y'|| playAgain=='Y');
    
    cout<<"Thanks for Playing Hangman!"<<endl;
    return 0;
}