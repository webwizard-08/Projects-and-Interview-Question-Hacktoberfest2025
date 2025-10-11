public class JavaInterviewDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Java Interview Questions Demo ===");
        
        demonstrateOOPConcepts();
        demonstrateCollections();
        demonstrateExceptionHandling();
        demonstrateStringManipulation();
        demonstrateArrayOperations();
    }
    
    public static void demonstrateOOPConcepts() {
        System.out.println("\n--- OOP Concepts Demo ---");
        
        Animal animal = new Dog("Buddy");
        animal.makeSound();
        animal.eat();
        
        System.out.println("Animal name: " + animal.getName());
        System.out.println("Is animal instance of Dog: " + (animal instanceof Dog));
    }
    
    public static void demonstrateCollections() {
        System.out.println("\n--- Collections Demo ---");
        
        java.util.List<String> list = new java.util.ArrayList<>();
        list.add("Java");
        list.add("Python");
        list.add("C++");
        
        System.out.println("List contents: " + list);
        System.out.println("List size: " + list.size());
        
        java.util.Set<Integer> set = new java.util.HashSet<>();
        set.add(1);
        set.add(2);
        set.add(1);
        System.out.println("Set contents (no duplicates): " + set);
        
        java.util.Map<String, Integer> map = new java.util.HashMap<>();
        map.put("Java", 95);
        map.put("Python", 90);
        map.put("C++", 85);
        System.out.println("Map contents: " + map);
    }
    
    public static void demonstrateExceptionHandling() {
        System.out.println("\n--- Exception Handling Demo ---");
        
        try {
            int result = divide(10, 0);
            System.out.println("Result: " + result);
        } catch (ArithmeticException e) {
            System.out.println("Caught exception: " + e.getMessage());
        } finally {
            System.out.println("Finally block executed");
        }
        
        try {
            String str = null;
            System.out.println("String length: " + str.length());
        } catch (NullPointerException e) {
            System.out.println("Caught NullPointerException: " + e.getMessage());
        }
    }
    
    public static void demonstrateStringManipulation() {
        System.out.println("\n--- String Manipulation Demo ---");
        
        String str1 = "Hello";
        String str2 = "World";
        String str3 = str1 + " " + str2;
        
        System.out.println("Concatenated string: " + str3);
        System.out.println("String length: " + str3.length());
        System.out.println("Uppercase: " + str3.toUpperCase());
        System.out.println("Contains 'World': " + str3.contains("World"));
        
        StringBuilder sb = new StringBuilder();
        sb.append("StringBuilder ");
        sb.append("is ");
        sb.append("mutable");
        System.out.println("StringBuilder result: " + sb.toString());
    }
    
    public static void demonstrateArrayOperations() {
        System.out.println("\n--- Array Operations Demo ---");
        
        int[] numbers = {5, 2, 8, 1, 9};
        System.out.println("Original array: " + java.util.Arrays.toString(numbers));
        
        java.util.Arrays.sort(numbers);
        System.out.println("Sorted array: " + java.util.Arrays.toString(numbers));
        
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        System.out.println("Sum of array elements: " + sum);
        System.out.println("Average: " + (double) sum / numbers.length);
    }
    
    public static int divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Division by zero is not allowed");
        }
        return a / b;
    }
}

abstract class Animal {
    protected String name;
    
    public Animal(String name) {
        this.name = name;
    }
    
    public abstract void makeSound();
    
    public void eat() {
        System.out.println(name + " is eating");
    }
    
    public String getName() {
        return name;
    }
}

class Dog extends Animal {
    public Dog(String name) {
        super(name);
    }
    
    @Override
    public void makeSound() {
        System.out.println(name + " barks: Woof! Woof!");
    }
    
    @Override
    public void eat() {
        System.out.println(name + " is eating dog food");
    }
}
