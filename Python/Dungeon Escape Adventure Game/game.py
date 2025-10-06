import random
import time
import sys

class Colors:
    """ANSI color codes for colorful output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.inventory = []
        self.keys = 0
        self.coins = 0
        self.strength = 10
        
    def add_item(self, item):
        self.inventory.append(item)
        print(f"{Colors.GREEN}✓ Added {item} to inventory!{Colors.RESET}")
        
    def show_inventory(self):
        if not self.inventory:
            print(f"{Colors.YELLOW}Your inventory is empty.{Colors.RESET}")
        else:
            print(f"\n{Colors.CYAN}╔═══════ INVENTORY ═══════╗{Colors.RESET}")
            for i, item in enumerate(self.inventory, 1):
                print(f"{Colors.CYAN}║{Colors.RESET} {i}. {item}")
            print(f"{Colors.CYAN}╚═════════════════════════╝{Colors.RESET}")
            
    def show_stats(self):
        print(f"\n{Colors.BOLD}╔════════ {self.name}'S STATS ════════╗{Colors.RESET}")
        print(f"║ Health: {Colors.GREEN}{'❤' * (self.health // 10)}{Colors.RESET}")
        print(f"║ HP: {self.health}/100")
        print(f"║ Strength: {self.strength}")
        print(f"║ Keys: {self.keys} 🗝")
        print(f"║ Coins: {self.coins} 💰")
        print(f"╚{'═' * 35}╝")

def slow_print(text, delay=0.03):
    """Print text with typing effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_banner():
    banner = f"""{Colors.BOLD}{Colors.RED}
╔══════════════════════════════════════════════╗
║                                              ║
║        🏰 DUNGEON ESCAPE ADVENTURE 🏰        ║
║                                              ║
║     Escape the cursed dungeon alive...      ║
║        or become trapped forever!           ║
║                                              ║
╚══════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)

def room_art(room_type):
    """ASCII art for different rooms"""
    art = {
        'entrance': f"""{Colors.YELLOW}
    🏰
   /||\\
  / || \\
 |  ||  |
 |______|
{Colors.RESET}""",
        'treasure': f"""{Colors.YELLOW}
    💎
   /|\\
  / | \\
 |  |  |
 |_____|
  Chest
{Colors.RESET}""",
        'monster': f"""{Colors.RED}
   👹
  /|\\
   |
  / \\
 MONSTER
{Colors.RESET}""",
        'puzzle': f"""{Colors.CYAN}
   ❓
  /|\\
 | ? |
  \\|/
 PUZZLE
{Colors.RESET}""",
        'exit': f"""{Colors.GREEN}
   🚪
  /||\\
 | || |
 | || |
 |____|
  EXIT
{Colors.RESET}"""
    }
    return art.get(room_type, "")

def entrance_room(player):
    """Starting room"""
    print(room_art('entrance'))
    slow_print(f"\n{Colors.CYAN}You wake up in a dark dungeon...{Colors.RESET}")
    slow_print("The walls are damp and covered in moss.")
    slow_print(f"You see {Colors.YELLOW}three doors{Colors.RESET} ahead of you.")
    
    print("\nWhich door do you choose?")
    print("1. Wooden door (looks old)")
    print("2. Iron door (looks sturdy)")
    print("3. Stone door (has ancient symbols)")
    
    choice = input(f"\n{Colors.YELLOW}Enter 1, 2, or 3: {Colors.RESET}")
    
    if choice == "1":
        treasure_room(player)
    elif choice == "2":
        monster_room(player)
    elif choice == "3":
        puzzle_room(player)
    else:
        print(f"{Colors.RED}Invalid choice! Try again.{Colors.RESET}")
        entrance_room(player)

def treasure_room(player):
    """Treasure room with choices"""
    print("\n" + "="*50)
    print(room_art('treasure'))
    slow_print(f"\n{Colors.YELLOW}You enter a room filled with treasures!{Colors.RESET}")
    
    if random.random() < 0.5:
        print(f"\n{Colors.GREEN}You found a golden key! 🗝{Colors.RESET}")
        player.keys += 1
    else:
        coins = random.randint(10, 50)
        print(f"\n{Colors.YELLOW}You found {coins} coins! 💰{Colors.RESET}")
        player.coins += coins
    
    print("\nWhat do you do?")
    print("1. Search for more treasure")
    print("2. Take the treasure and move on")
    print("3. Check inventory")
    
    choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.RESET}")
    
    if choice == "1":
        if random.random() < 0.3:
            print(f"\n{Colors.GREEN}Lucky! You found a health potion!{Colors.RESET}")
            player.add_item("Health Potion")
        else:
            damage = random.randint(10, 20)
            player.health -= damage
            print(f"\n{Colors.RED}TRAP! You triggered a spike trap!{Colors.RESET}")
            print(f"{Colors.RED}Lost {damage} HP!{Colors.RESET}")
    elif choice == "3":
        player.show_inventory()
    
    player.show_stats()
    
    if player.health > 0:
        corridor_choice(player)

def monster_room(player):
    """Combat encounter"""
    print("\n" + "="*50)
    print(room_art('monster'))
    slow_print(f"\n{Colors.RED}A wild monster appears!{Colors.RESET}")
    
    monster_health = random.randint(30, 50)
    monster_attack = random.randint(5, 15)
    
    print(f"\n{Colors.RED}Monster HP: {monster_health}{Colors.RESET}")
    
    while monster_health > 0 and player.health > 0:
        print(f"\n{Colors.BOLD}Your turn!{Colors.RESET}")
        print("1. Attack")
        print("2. Use item")
        print("3. Run away")
        
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.RESET}")
        
        if choice == "1":
            damage = player.strength + random.randint(5, 15)
            monster_health -= damage
            print(f"\n{Colors.GREEN}You hit for {damage} damage!{Colors.RESET}")
            
            if monster_health > 0:
                print(f"{Colors.RED}Monster HP: {monster_health}{Colors.RESET}")
                
                # Monster attacks back
                monster_damage = monster_attack + random.randint(0, 10)
                player.health -= monster_damage
                print(f"{Colors.RED}Monster attacks for {monster_damage} damage!{Colors.RESET}")
                print(f"{Colors.YELLOW}Your HP: {player.health}{Colors.RESET}")
            else:
                print(f"\n{Colors.GREEN}💀 Monster defeated!{Colors.RESET}")
                reward = random.randint(20, 40)
                player.coins += reward
                print(f"{Colors.YELLOW}Found {reward} coins on the monster!{Colors.RESET}")
                
        elif choice == "2":
            player.show_inventory()
            if "Health Potion" in player.inventory:
                use = input("Use Health Potion? (y/n): ")
                if use.lower() == 'y':
                    heal = random.randint(20, 40)
                    player.health = min(100, player.health + heal)
                    player.inventory.remove("Health Potion")
                    print(f"{Colors.GREEN}Healed {heal} HP!{Colors.RESET}")
        elif choice == "3":
            if random.random() < 0.5:
                print(f"\n{Colors.GREEN}You escaped!{Colors.RESET}")
                corridor_choice(player)
                return
            else:
                print(f"\n{Colors.RED}Failed to escape!{Colors.RESET}")
                damage = monster_attack
                player.health -= damage
                print(f"{Colors.RED}Monster hits you for {damage} damage!{Colors.RESET}")
    
    if player.health <= 0:
        game_over(player)
    else:
        player.show_stats()
        corridor_choice(player)

def puzzle_room(player):
    """Puzzle challenge"""
    print("\n" + "="*50)
    print(room_art('puzzle'))
    slow_print(f"\n{Colors.CYAN}You enter a mysterious room with ancient inscriptions...{Colors.RESET}")
    
    puzzles = [
        {
            "question": "I speak without a mouth and hear without ears. I have no body, but come alive with wind. What am I?",
            "answer": "echo",
            "hint": "Think about sound..."
        },
        {
            "question": "What has keys but no locks, space but no room, and you can enter but can't go inside?",
            "answer": "keyboard",
            "hint": "You're probably using one right now..."
        },
        {
            "question": "The more you take, the more you leave behind. What am I?",
            "answer": "footsteps",
            "hint": "Think about walking..."
        }
    ]
    
    puzzle = random.choice(puzzles)
    
    print(f"\n{Colors.YELLOW}The inscription reads:{Colors.RESET}")
    slow_print(f'"{puzzle["question"]}"')
    
    print("\n1. Answer the riddle")
    print("2. Ask for hint")
    print("3. Skip the puzzle")
    
    choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.RESET}")
    
    if choice == "1":
        answer = input("\nYour answer: ").lower().strip()
        if answer == puzzle["answer"]:
            print(f"\n{Colors.GREEN}✓ Correct! The door opens...{Colors.RESET}")
            player.keys += 1
            print(f"{Colors.GREEN}You found a golden key!{Colors.RESET}")
            player.strength += 5
            print(f"{Colors.GREEN}Your strength increased by 5!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}✗ Wrong answer!{Colors.RESET}")
            print(f"The correct answer was: {Colors.CYAN}{puzzle['answer']}{Colors.RESET}")
            damage = 15
            player.health -= damage
            print(f"{Colors.RED}Magic trap activated! Lost {damage} HP!{Colors.RESET}")
    elif choice == "2":
        print(f"\n{Colors.CYAN}Hint: {puzzle['hint']}{Colors.RESET}")
        answer = input("\nYour answer: ").lower().strip()
        if answer == puzzle["answer"]:
            print(f"\n{Colors.GREEN}✓ Correct!{Colors.RESET}")
            player.keys += 1
        else:
            print(f"\n{Colors.RED}✗ Still wrong!{Colors.RESET}")
            player.health -= 10
    
    player.show_stats()
    
    if player.health > 0:
        corridor_choice(player)

def corridor_choice(player):
    """Corridor with multiple paths"""
    print("\n" + "="*50)
    slow_print(f"\n{Colors.CYAN}You're in a long corridor...{Colors.RESET}")
    
    if player.keys >= 2:
        print(f"\n{Colors.GREEN}You have enough keys to unlock the exit!{Colors.RESET}")
        print("1. Go to the exit")
        print("2. Explore more rooms")
        
        choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.RESET}")
        if choice == "1":
            exit_room(player)
            return
    
    print("\nWhich path do you take?")
    print("1. Left path (you hear water dripping)")
    print("2. Right path (you see a faint light)")
    print("3. Straight ahead (dark and quiet)")
    
    choice = input(f"\n{Colors.YELLOW}Your choice: {Colors.RESET}")
    
    paths = [treasure_room, monster_room, puzzle_room]
    
    if choice in ["1", "2", "3"]:
        random.choice(paths)(player)
    else:
        print(f"{Colors.RED}Invalid choice!{Colors.RESET}")
        corridor_choice(player)

def exit_room(player):
    """Final room - escape!"""
    print("\n" + "="*50)
    print(room_art('exit'))
    slow_print(f"\n{Colors.GREEN}You found the exit!{Colors.RESET}")
    
    if player.keys >= 2:
        slow_print(f"{Colors.BOLD}{Colors.GREEN}You use your keys to unlock the door...{Colors.RESET}")
        time.sleep(1)
        slow_print("The door creaks open...")
        time.sleep(1)
        slow_print("FREEDOM!")
        victory(player)
    else:
        print(f"\n{Colors.RED}You need at least 2 keys to unlock this door!{Colors.RESET}")
        print(f"You currently have {player.keys} key(s).")
        slow_print("You must explore more...")
        corridor_choice(player)

def victory(player):
    """Victory screen"""
    victory_art = f"""{Colors.GREEN}{Colors.BOLD}
╔═══════════════════════════════════╗
║                                   ║
║          🎉 VICTORY! 🎉          ║
║                                   ║
║   You escaped the dungeon alive!  ║
║                                   ║
╚═══════════════════════════════════╝
{Colors.RESET}"""
    
    print(victory_art)
    print(f"\n{Colors.BOLD}Final Stats:{Colors.RESET}")
    print(f"Health remaining: {Colors.GREEN}{player.health}/100{Colors.RESET}")
    print(f"Coins collected: {Colors.YELLOW}{player.coins} 💰{Colors.RESET}")
    print(f"Keys found: {Colors.CYAN}{player.keys} 🗝{Colors.RESET}")
    print(f"\n{Colors.GREEN}Congratulations, {player.name}!{Colors.RESET}")

def game_over(player):
    """Game over screen"""
    game_over_art = f"""{Colors.RED}{Colors.BOLD}
╔═══════════════════════════════════╗
║                                   ║
║         💀 GAME OVER 💀          ║
║                                   ║
║    You died in the dungeon...     ║
║                                   ║
╚═══════════════════════════════════╝
{Colors.RESET}"""
    
    print(game_over_art)
    print(f"\nYou survived: {Colors.YELLOW}Several rooms{Colors.RESET}")
    print(f"Coins collected: {Colors.YELLOW}{player.coins} 💰{Colors.RESET}")
    print(f"\n{Colors.CYAN}Better luck next time, {player.name}!{Colors.RESET}")

def main():
    """Main game function"""
    print_banner()
    
    print(f"\n{Colors.CYAN}Welcome, brave adventurer!{Colors.RESET}")
    name = input(f"{Colors.YELLOW}Enter your name: {Colors.RESET}").strip() or "Hero"
    
    player = Player(name)
    
    slow_print(f"\n{Colors.GREEN}Welcome, {name}!{Colors.RESET}")
    slow_print("Your quest: Escape the cursed dungeon!")
    slow_print("Find keys, avoid traps, and survive monsters...")
    
    input(f"\n{Colors.YELLOW}Press Enter to begin your adventure...{Colors.RESET}")
    
    entrance_room(player)

if __name__ == "__main__":
    main()
