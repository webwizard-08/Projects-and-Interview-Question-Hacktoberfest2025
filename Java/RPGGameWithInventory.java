import java.util.*;

class Player {
    String name;
    int hp, maxHp, attack, defense, level, exp;
    Map<String, Integer> inventory = new HashMap<>();

    public Player(String name) {
        this.name = name;
        this.level = 1;
        this.exp = 0;
        this.maxHp = 50;
        this.hp = maxHp;
        this.attack = 10;
        this.defense = 5;
        addItem("Potion", 2); // start with 2 potions
    }

    public void levelUp() {
        level++;
        maxHp += 10;
        hp = maxHp;
        attack += 3;
        defense += 2;
        System.out.println("⚡ LEVEL UP! You are now level " + level + "!");
    }

    public void addItem(String item, int amount) {
        inventory.put(item, inventory.getOrDefault(item, 0) + amount);
    }

    public void useItem(String item) {
        if (!inventory.containsKey(item) || inventory.get(item) <= 0) {
            System.out.println("❌ You don’t have any " + item + "!");
            return;
        }

        switch (item) {
            case "Potion":
                if (hp < maxHp) {
                    hp = Math.min(maxHp, hp + 20);
                    System.out.println("🧪 You used a Potion! HP restored to " + hp + "/" + maxHp);
                } else {
                    System.out.println("HP is already full!");
                    return;
                }
                break;
            case "Elixir":
                hp = maxHp;
                System.out.println("✨ You used an Elixir! HP fully restored.");
                break;
            case "Sword":
                attack += 5;
                System.out.println("⚔️ You equipped a Sword! Attack increased to " + attack);
                break;
            case "Shield":
                defense += 5;
                System.out.println("🛡️ You equipped a Shield! Defense increased to " + defense);
                break;
            default:
                System.out.println("❓ Unknown item.");
                return;
        }
        // decrease quantity
        inventory.put(item, inventory.get(item) - 1);
        if (inventory.get(item) == 0) inventory.remove(item);
    }

    public void showInventory() {
        if (inventory.isEmpty()) {
            System.out.println("📦 Inventory is empty.");
        } else {
            System.out.println("📦 Inventory: " + inventory);
        }
    }
}

class Enemy {
    String name;
    int hp, attack, defense;

    public Enemy(String name, int hp, int attack, int defense) {
        this.name = name;
        this.hp = hp;
        this.attack = attack;
        this.defense = defense;
    }
}

public class RPGGameWithInventory {
    static Scanner scanner = new Scanner(System.in);
    static Random rand = new Random();

    public static void main(String[] args) {
        System.out.println("=== RPG TEXT ADVENTURE ===");
        System.out.print("Enter your hero's name: ");
        String name = scanner.nextLine();

        Player player = new Player(name);
        System.out.println("Welcome, " + player.name + "! Your adventure begins...\n");

        while (player.hp > 0) {
            Enemy enemy = generateEnemy();
            System.out.println("⚔️ A wild " + enemy.name + " appears! (HP: " + enemy.hp + ")\n");

            while (enemy.hp > 0 && player.hp > 0) {
                System.out.println(player.name + " (HP: " + player.hp + "/" + player.maxHp + ")");
                player.showInventory();
                System.out.println("1. Attack");
                System.out.println("2. Defend");
                System.out.println("3. Use Item");
                System.out.println("4. Run");
                System.out.print("Choose action: ");
                int choice = scanner.nextInt();

                if (choice == 1) { // attack
                    int damage = Math.max(0, player.attack - enemy.defense + rand.nextInt(5));
                    enemy.hp -= damage;
                    System.out.println("💥 You attack " + enemy.name + " and deal " + damage + " damage!");
                } else if (choice == 2) { // defend
                    System.out.println("🛡️ You defend! Defense increased temporarily.");
                    player.defense += 3;
                } else if (choice == 3) { // use item
                    scanner.nextLine(); // clear buffer
                    System.out.print("Enter item name to use: ");
                    String item = scanner.nextLine();
                    player.useItem(item);
                } else if (choice == 4) { // run
                    System.out.println("🏃 You escaped safely!");
                    break;
                }

                // enemy attack
                if (enemy.hp > 0) {
                    int damage = Math.max(0, enemy.attack - player.defense + rand.nextInt(3));
                    player.hp -= damage;
                    System.out.println("⚔️ " + enemy.name + " hits you for " + damage + " damage!");
                }

                if (choice == 2) {
                    player.defense -= 3; // reset defense buff
                }
                System.out.println();
            }

            if (player.hp <= 0) {
                System.out.println("💀 You were defeated... Game Over!");
                break;
            }

            if (enemy.hp <= 0) {
                System.out.println("🎉 You defeated " + enemy.name + "!");
                player.exp += 10;
                if (player.exp >= player.level * 20) {
                    player.exp = 0;
                    player.levelUp();
                }

                // random item drop
                if (rand.nextBoolean()) {
                    String[] items = {"Potion", "Elixir", "Sword", "Shield"};
                    String drop = items[rand.nextInt(items.length)];
                    player.addItem(drop, 1);
                    System.out.println("🎁 The enemy dropped a " + drop + "! It was added to your inventory.");
                }
            }

            System.out.println("\nContinue your adventure? (y/n): ");
            String cont = scanner.next();
            if (!cont.equalsIgnoreCase("y")) {
                System.out.println("👋 Thanks for playing!");
                break;
            }
        }
    }

    public static Enemy generateEnemy() {
        String[] names = {"Goblin", "Orc", "Wolf", "Troll", "Skeleton"};
        String name = names[rand.nextInt(names.length)];
        int hp = rand.nextInt(30) + 20;
        int attack = rand.nextInt(10) + 5;
        int defense = rand.nextInt(5) + 2;
        return new Enemy(name, hp, attack, defense);
    }
}￼Enter
