/*
 * 🧠 Top 20 Stack Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This Java program contains 20 essential stack problems frequently asked
 * in technical interviews at top companies like FAANG, TCS, Infosys, and Amazon.
 *
 * Each problem includes:
 *  - Problem definition (short)
 *  - Clean Java implementation (modular)
 *  - Example I/O (demonstrated in menu)
 *  - Time and Space Complexity (commented)
 */

import java.util.*;

public class Stack_Interview_Questions {

    /* ------------------------------
     * 1) Implement Stack using Array
     * Time: O(1) per op, Space: O(n)
     * ------------------------------ */
    static class ArrayStack {
        private int[] arr;
        private int top;
        ArrayStack(int capacity) {
            arr = new int[capacity];
            top = -1;
        }
        void push(int x) {
            if (top == arr.length - 1) throw new RuntimeException("Overflow");
            arr[++top] = x;
        }
        int pop() {
            if (top == -1) throw new RuntimeException("Underflow");
            return arr[top--];
        }
        int peek() {
            if (top == -1) throw new RuntimeException("Empty");
            return arr[top];
        }
        boolean isEmpty() { return top == -1; }
    }

    /* ------------------------------
     * 2) Implement Stack using Linked List
     * Time: O(1) per op, Space: O(n)
     * ------------------------------ */
    static class LinkedStack {
        static class Node { int val; Node next; Node(int v){val=v;} }
        private Node head;
        void push(int x) { Node n = new Node(x); n.next = head; head = n; }
        int pop() {
            if (head == null) throw new RuntimeException("Empty");
            int v = head.val; head = head.next; return v;
        }
        int peek() {
            if (head == null) throw new RuntimeException("Empty");
            return head.val;
        }
        boolean isEmpty() { return head == null; }
    }

    /* ------------------------------
     * 3) Reverse a String using Stack
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static String reverseStringUsingStack(String s) {
        Stack<Character> st = new Stack<>();
        for (char c : s.toCharArray()) st.push(c);
        StringBuilder sb = new StringBuilder();
        while (!st.isEmpty()) sb.append(st.pop());
        return sb.toString();
    }

    /* ------------------------------
     * 4) Check for Balanced Parentheses
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static boolean isBalanced(String s) {
        Stack<Character> st = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (ch == '(' || ch == '{' || ch == '[') st.push(ch);
            else if (ch == ')' || ch == '}' || ch == ']') {
                if (st.isEmpty()) return false;
                char top = st.pop();
                if ((ch == ')' && top != '(') ||
                    (ch == '}' && top != '{') ||
                    (ch == ']' && top != '[')) return false;
            }
        }
        return st.isEmpty();
    }

    /* ------------------------------
     * 5) Evaluate Postfix Expression
     * e.g. "2 3 1 * + 9 -" => result
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int evalPostfix(String expr) {
        Stack<Integer> st = new Stack<>();
        String[] tokens = expr.trim().split("\\s+");
        for (String t : tokens) {
            if (t.matches("-?\\d+")) st.push(Integer.parseInt(t));
            else {
                int b = st.pop();
                int a = st.pop();
                switch (t) {
                    case "+": st.push(a + b); break;
                    case "-": st.push(a - b); break;
                    case "*": st.push(a * b); break;
                    case "/": st.push(a / b); break;
                    case "^": st.push((int)Math.pow(a, b)); break;
                    default: throw new RuntimeException("Unknown op " + t);
                }
            }
        }
        return st.pop();
    }

    /* ------------------------------
     * 6) Infix to Postfix (Shunting-yard)
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    private static int prec(char c) {
        switch (c) {
            case '+': case '-': return 1;
            case '*': case '/': return 2;
            case '^': return 3;
            default: return -1;
        }
    }
    public static String infixToPostfix(String s) {
        StringBuilder out = new StringBuilder();
        Stack<Character> st = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isWhitespace(c)) continue;
            if (Character.isLetterOrDigit(c)) out.append(c);
            else if (c == '(') st.push(c);
            else if (c == ')') {
                while (!st.isEmpty() && st.peek() != '(') out.append(' ').append(st.pop());
                if (!st.isEmpty() && st.peek() == '(') st.pop();
            } else { // operator
                out.append(' ');
                while (!st.isEmpty() && prec(c) <= prec(st.peek()) && st.peek() != '(') out.append(st.pop()).append(' ');
                st.push(c);
            }
        }
        while (!st.isEmpty()) out.append(' ').append(st.pop());
        return out.toString().trim().replaceAll("\\s+", " ");
    }

    /* ------------------------------
     * 7) Next Greater Element (to right)
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int[] nextGreaterElement(int[] arr) {
        int n = arr.length;
        int[] res = new int[n];
        Stack<Integer> st = new Stack<>(); // stores indices
        for (int i = n - 1; i >= 0; i--) {
            while (!st.isEmpty() && arr[st.peek()] <= arr[i]) st.pop();
            res[i] = st.isEmpty() ? -1 : arr[st.peek()];
            st.push(i);
        }
        return res;
    }

    /* ------------------------------
     * 8) Next Smaller Element (to right)
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int[] nextSmallerElement(int[] arr) {
        int n = arr.length;
        int[] res = new int[n];
        Stack<Integer> st = new Stack<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!st.isEmpty() && st.peek() >= arr[i]) st.pop();
            res[i] = st.isEmpty() ? -1 : st.peek();
            st.push(arr[i]);
        }
        return res;
    }

    /* ------------------------------
     * 9) Stock Span Problem
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int[] stockSpan(int[] prices) {
        int n = prices.length;
        int[] span = new int[n];
        Stack<Integer> st = new Stack<>(); // indices
        for (int i = 0; i < n; i++) {
            while (!st.isEmpty() && prices[st.peek()] <= prices[i]) st.pop();
            span[i] = st.isEmpty() ? i + 1 : i - st.peek();
            st.push(i);
        }
        return span;
    }

    /* ------------------------------
     * 10) Min Element in Stack with O(1) extra space
     * Technique: store encoded values
     * Time: O(1) per op, Space: O(n) (only stack)
     * ------------------------------ */
    static class MinStack {
        private Stack<Long> st = new Stack<>();
        private long min;
        void push(int x) {
            if (st.isEmpty()) { st.push((long)x); min = x; }
            else {
                if (x < min) {
                    // encode
                    st.push(2L * x - min);
                    min = x;
                } else st.push((long)x);
            }
        }
        int pop() {
            if (st.isEmpty()) throw new RuntimeException("Empty");
            long t = st.pop();
            if (t < min) {
                long oldMin = min;
                min = 2 * min - t; // decode previous min
                return (int)oldMin;
            } else return (int)t;
        }
        int getMin() {
            if (st.isEmpty()) throw new RuntimeException("Empty");
            return (int)min;
        }
    }

    /* ------------------------------
     * 11) Sort a Stack using Recursion
     * Time: O(n^2), Space: O(n)
     * ------------------------------ */
    public static void sortStackRec(Stack<Integer> s) {
        if (s.isEmpty()) return;
        int x = s.pop();
        sortStackRec(s);
        insertSorted(s, x);
    }
    private static void insertSorted(Stack<Integer> s, int x) {
        if (s.isEmpty() || s.peek() <= x) { s.push(x); return; }
        int t = s.pop();
        insertSorted(s, x);
        s.push(t);
    }

    /* ------------------------------
     * 12) Reverse a Stack using Recursion
     * Time: O(n^2) worst, Space: O(n)
     * ------------------------------ */
    public static void reverseStackRec(Stack<Integer> s) {
        if (s.isEmpty()) return;
        int x = s.pop();
        reverseStackRec(s);
        insertAtBottom(s, x);
    }
    private static void insertAtBottom(Stack<Integer> s, int x) {
        if (s.isEmpty()) { s.push(x); return; }
        int t = s.pop();
        insertAtBottom(s, x);
        s.push(t);
    }

    /* ------------------------------
     * 13) Celebrity Problem (using stack)
     * Input: matrix M where M[a][b] == 1 means a knows b
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int findCelebrity(int[][] M) {
        int n = M.length;
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < n; i++) st.push(i);
        while (st.size() > 1) {
            int a = st.pop(), b = st.pop();
            if (knows(M, a, b)) st.push(b); else st.push(a);
        }
        if (st.isEmpty()) return -1;
        int cand = st.pop();
        for (int i = 0; i < n; i++) {
            if (i == cand) continue;
            if (knows(M, cand, i) || !knows(M, i, cand)) return -1;
        }
        return cand;
    }
    private static boolean knows(int[][] M, int a, int b) { return M[a][b] == 1; }

    /* ------------------------------
     * 14) Largest Rectangle in Histogram
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int largestRectangleHistogram(int[] h) {
        int n = h.length;
        Stack<Integer> st = new Stack<>();
        int maxA = 0;
        for (int i = 0; i <= n; i++) {
            int curr = (i == n) ? 0 : h[i];
            while (!st.isEmpty() && curr < h[st.peek()]) {
                int height = h[st.pop()];
                int left = st.isEmpty() ? 0 : st.peek() + 1;
                int width = i - left;
                maxA = Math.max(maxA, height * width);
            }
            st.push(i);
        }
        return maxA;
    }

    /* ------------------------------
     * 15) Trapping Rain Water (stack approach)
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int trapRainWater(int[] height) {
        int n = height.length;
        Stack<Integer> st = new Stack<>();
        int i = 0, water = 0;
        while (i < n) {
            while (!st.isEmpty() && height[i] > height[st.peek()]) {
                int top = st.pop();
                if (st.isEmpty()) break;
                int distance = i - st.peek() - 1;
                int bounded = Math.min(height[i], height[st.peek()]) - height[top];
                water += distance * bounded;
            }
            st.push(i++);
        }
        return water;
    }

    /* ------------------------------
     * 16) Remove Consecutive Duplicates (using stack for string)
     * e.g. "aabbcc" -> "abc" or if want to remove adjacent duplicates fully "abbaca" -> ?
     * We'll remove immediate adjacent duplicates leaving single char: "aaab" -> "ab"
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static String removeConsecutiveDuplicates(String s) {
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (sb.length() > 0 && sb.charAt(sb.length()-1) == c) continue;
            sb.append(c);
        }
        return sb.toString();
    }

    /* ------------------------------
     * 17) Valid Parenthesis String with '*' (LeetCode 678)
     * '*' can be '(', ')' or empty
     * Time: O(n), Space: O(1)
     * ------------------------------ */
    public static boolean checkValidString(String s) {
        int lo = 0, hi = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') { lo++; hi++; }
            else if (c == ')') { lo = Math.max(lo - 1, 0); hi--; }
            else { // *
                lo = Math.max(lo - 1, 0);
                hi++;
            }
            if (hi < 0) return false;
        }
        return lo == 0;
    }

    /* ------------------------------
     * 18) Decode String (e.g., "3[a2[b]]" -> "abbabbabb")
     * Time: O(n * k) where k = repeat counts, Space: O(n)
     * ------------------------------ */
    public static String decodeString(String s) {
        Stack<Integer> counts = new Stack<>();
        Stack<StringBuilder> resultStack = new Stack<>();
        StringBuilder current = new StringBuilder();
        int k = 0;
        for (char ch : s.toCharArray()) {
            if (Character.isDigit(ch)) { k = k * 10 + (ch - '0'); }
            else if (ch == '[') {
                counts.push(k); k = 0;
                resultStack.push(current);
                current = new StringBuilder();
            } else if (ch == ']') {
                StringBuilder tmp = resultStack.pop();
                int repeat = counts.pop();
                for (int i = 0; i < repeat; i++) tmp.append(current);
                current = tmp;
            } else current.append(ch);
        }
        return current.toString();
    }

    /* ------------------------------
     * 19) Nearest Greater to Left (NGL)
     * Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int[] nearestGreaterLeft(int[] arr) {
        int n = arr.length; int[] res = new int[n];
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < n; i++) {
            while (!st.isEmpty() && st.peek() <= arr[i]) st.pop();
            res[i] = st.isEmpty() ? -1 : st.peek();
            st.push(arr[i]);
        }
        return res;
    }

    /* ------------------------------
     * 20) Evaluate Infix Expression (convert to postfix then eval)
     * Supports + - * / ^ and parentheses. Time: O(n), Space: O(n)
     * ------------------------------ */
    public static int evalInfix(String expr) {
        String postfix = infixToPostfix(expr);
        // Convert postfix tokens into spaced tokens (if letters/digits were single char)
        // For simplicity assume single-digit numbers or no spaces => we'll split by spaces after conversion
        return evalPostfix(postfix.replaceAll("(?<=\\d)(?=\\D)|(?<=\\D)(?=\\d)").replaceAll("\\s+", " ").trim());
    }

    /* ------------------------------
     * Utility: print array
     * ------------------------------ */
    private static void printArray(int[] arr) {
        System.out.println(Arrays.toString(arr));
    }

    /* ------------------------------
     * Menu-driven main
     * ------------------------------ */
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println("\n=== Stack Interview Questions Menu ===");
            System.out.println("1  - Array Stack (demo)");
            System.out.println("2  - LinkedList Stack (demo)");
            System.out.println("3  - Reverse String using Stack");
            System.out.println("4  - Balanced Parentheses");
            System.out.println("5  - Evaluate Postfix");
            System.out.println("6  - Infix to Postfix");
            System.out.println("7  - Next Greater Element");
            System.out.println("8  - Next Smaller Element");
            System.out.println("9  - Stock Span Problem");
            System.out.println("10 - Min Stack (O(1) getMin)");
            System.out.println("11 - Sort Stack (recursion)");
            System.out.println("12 - Reverse Stack (recursion)");
            System.out.println("13 - Celebrity Problem");
            System.out.println("14 - Largest Rectangle in Histogram");
            System.out.println("15 - Trapping Rain Water");
            System.out.println("16 - Remove Consecutive Duplicates (string)");
            System.out.println("17 - Valid Parenthesis String with '*'");
            System.out.println("18 - Decode String (k[encoded])");
            System.out.println("19 - Nearest Greater to Left");
            System.out.println("20 - Evaluate Infix Expression");
            System.out.println("0  - Exit");
            System.out.print("Choose option: ");
            int choice;
            try { choice = Integer.parseInt(sc.nextLine().trim()); }
            catch (Exception e) { System.out.println("Invalid input"); continue; }

            switch (choice) {
                case 0: System.out.println("Bye!"); sc.close(); return;
                case 1: {
                    ArrayStack st = new ArrayStack(5);
                    st.push(10); st.push(20); st.push(30);
                    System.out.println("Popped: " + st.pop() + " Peek: " + st.peek());
                    break;
                }
                case 2: {
                    LinkedStack s = new LinkedStack();
                    s.push(5); s.push(7); s.push(9);
                    System.out.println("Popped: " + s.pop() + " Peek: " + s.peek());
                    break;
                }
                case 3: {
                    System.out.print("Enter string: ");
                    String s = sc.nextLine();
                    System.out.println("Reversed: " + reverseStringUsingStack(s));
                    break;
                }
                case 4: {
                    System.out.print("Enter expression: ");
                    String s = sc.nextLine();
                    System.out.println(isBalanced(s) ? "Balanced" : "Not Balanced");
                    break;
                }
                case 5: {
                    System.out.print("Enter postfix (space separated): ");
                    String expr = sc.nextLine();
                    System.out.println("Result: " + evalPostfix(expr));
                    break;
                }
                case 6: {
                    System.out.print("Enter infix (single-letter operands or spaced): ");
                    String inf = sc.nextLine();
                    System.out.println("Postfix: " + infixToPostfix(inf));
                    break;
                }
                case 7: {
                    int[] arr = {4, 5, 2, 25};
                    System.out.println("Arr: " + Arrays.toString(arr));
                    System.out.print("Next Greater: ");
                    printArray(nextGreaterElement(arr));
                    break;
                }
                case 8: {
                    int[] arr = {4, 5, 2, 25};
                    System.out.println("Arr: " + Arrays.toString(arr));
                    System.out.print("Next Smaller: ");
                    printArray(nextSmallerElement(arr));
                    break;
                }
                case 9: {
                    int[] prices = {100, 80, 60, 70, 60, 75, 85};
                    System.out.println("Prices: " + Arrays.toString(prices));
                    System.out.print("Span: ");
                    printArray(stockSpan(prices));
                    break;
                }
                case 10: {
                    MinStack ms = new MinStack();
                    ms.push(3); ms.push(5); System.out.println("Min: " + ms.getMin());
                    ms.push(2); ms.push(1); System.out.println("Min after pushes: " + ms.getMin());
                    ms.pop(); System.out.println("Min after pop: " + ms.getMin());
                    break;
                }
                case 11: {
                    Stack<Integer> srt = new Stack<>();
                    srt.push(30); srt.push(-5); srt.push(18); srt.push(14); srt.push(-3);
                    System.out.println("Before: " + srt);
                    sortStackRec(srt);
                    System.out.println("After: " + srt);
                    break;
                }
                case 12: {
                    Stack<Integer> srev = new Stack<>();
                    srev.push(1); srev.push(2); srev.push(3); srev.push(4);
                    System.out.println("Before: " + srev);
                    reverseStackRec(srev);
                    System.out.println("After: " + srev);
                    break;
                }
                case 13: {
                    // sample matrix where 1 = knows; celebrity = index 1
                    int[][] M = {
                        {0,1,0},
                        {0,0,0},
                        {0,1,0}
                    };
                    int celeb = findCelebrity(M);
                    System.out.println("Celebrity index: " + celeb);
                    break;
                }
                case 14: {
                    int[] hist = {2,1,5,6,2,3};
                    System.out.println("Histogram: " + Arrays.toString(hist));
                    System.out.println("Largest Rect Area: " + largestRectangleHistogram(hist));
                    break;
                }
                case 15: {
                    int[] h = {0,1,0,2,1,0,1,3,2,1,2,1};
                    System.out.println("Heights: " + Arrays.toString(h));
                    System.out.println("Trapped Water: " + trapRainWater(h));
                    break;
                }
                case 16: {
                    System.out.print("Enter string: ");
                    String str = sc.nextLine();
                    System.out.println("Result: " + removeConsecutiveDuplicates(str));
                    break;
                }
                case 17: {
                    System.out.print("Enter string with *: ");
                    String str = sc.nextLine();
                    System.out.println(checkValidString(str) ? "Valid" : "Invalid");
                    break;
                }
                case 18: {
                    System.out.print("Enter encoded string (e.g., 3[a2[b]]): ");
                    String enc = sc.nextLine();
                    System.out.println("Decoded: " + decodeString(enc));
                    break;
                }
                case 19: {
                    int[] arr = {1, 3, 2, 4};
                    System.out.println("Arr: " + Arrays.toString(arr));
                    System.out.print("Nearest Greater to Left: ");
                    printArray(nearestGreaterLeft(arr));
                    break;
                }
                case 20: {
                    System.out.print("Enter infix expression (space separated or single-digit nums): ");
                    String inf = sc.nextLine();
                    // convert and evaluate — this evalInfix uses a simple approach expecting spaced tokens
                    try {
                        String postfix = infixToPostfix(inf);
                        System.out.println("Postfix: " + postfix);
                        System.out.println("Eval: " + evalPostfix(postfix));
                    } catch (Exception e) {
                        System.out.println("Error evaluating expression: " + e.getMessage());
                    }
                    break;
                }
                default:
                    System.out.println("Invalid option.");
            }
        }
    }
}
