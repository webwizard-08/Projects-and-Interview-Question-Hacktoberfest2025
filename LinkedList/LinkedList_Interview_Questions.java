/*
 * 🧠 Top 20 Linked List Interview Questions – Java Implementation
 * Author: Sai Surya
 *
 * Description:
 * This Java program contains 20 essential linked list problems frequently asked
 * in technical interviews at top companies like FAANG, TCS, Infosys, and Amazon.
 *
 * Each problem includes:
 *  - Problem definition
 *  - Clean Java implementation
 *  - Example I/O
 *  - Time and Space Complexity
 */

import java.util.*;

class LinkedListNode {
    int data;
    LinkedListNode next;
    LinkedListNode random; // for random pointer problem
    LinkedListNode(int data) {
        this.data = data;
        this.next = null;
        this.random = null;
    }
}

public class LinkedList_Interview_Questions {

    // 1️⃣ Insert at Head
    public static LinkedListNode insertAtHead(LinkedListNode head, int data) {
        LinkedListNode newNode = new LinkedListNode(data);
        newNode.next = head;
        return newNode; // O(1)
    }

    // 2️⃣ Insert at Tail
    public static LinkedListNode insertAtTail(LinkedListNode head, int data) {
        LinkedListNode newNode = new LinkedListNode(data);
        if (head == null) return newNode;
        LinkedListNode curr = head;
        while (curr.next != null) curr = curr.next;
        curr.next = newNode;
        return head; // O(n)
    }

    // 3️⃣ Delete a Node by Key
    public static LinkedListNode deleteNode(LinkedListNode head, int key) {
        if (head == null) return null;
        if (head.data == key) return head.next;
        LinkedListNode curr = head;
        while (curr.next != null && curr.next.data != key) curr = curr.next;
        if (curr.next != null) curr.next = curr.next.next;
        return head; // O(n)
    }

    // 4️⃣ Reverse Linked List Iteratively
    public static LinkedListNode reverseIterative(LinkedListNode head) {
        LinkedListNode prev = null, curr = head;
        while (curr != null) {
            LinkedListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev; // O(n)
    }

    // 5️⃣ Reverse Linked List Recursively
    public static LinkedListNode reverseRecursive(LinkedListNode head) {
        if (head == null || head.next == null) return head;
        LinkedListNode rest = reverseRecursive(head.next);
        head.next.next = head;
        head.next = null;
        return rest; // O(n)
    }

    // 6️⃣ Find Middle Node
    public static LinkedListNode findMiddle(LinkedListNode head) {
        LinkedListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow; // O(n)
    }

    // 7️⃣ Detect Loop (Floyd’s Cycle Detection)
    public static boolean detectLoop(LinkedListNode head) {
        LinkedListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false; // O(n)
    }

    // 8️⃣ Remove Duplicates from Sorted Linked List
    public static LinkedListNode removeDuplicates(LinkedListNode head) {
        LinkedListNode curr = head;
        while (curr != null && curr.next != null) {
            if (curr.data == curr.next.data) curr.next = curr.next.next;
            else curr = curr.next;
        }
        return head; // O(n)
    }

    // 9️⃣ Check if Palindrome
    public static boolean isPalindrome(LinkedListNode head) {
        LinkedListNode slow = head, fast = head;
        Stack<Integer> stack = new Stack<>();
        while (fast != null && fast.next != null) {
            stack.push(slow.data);
            slow = slow.next;
            fast = fast.next.next;
        }
        if (fast != null) slow = slow.next; // odd number of elements
        while (slow != null) {
            if (stack.pop() != slow.data) return false;
            slow = slow.next;
        }
        return true; // O(n), O(n) space
    }

    // 🔟 Nth Node from End
    public static LinkedListNode nthFromEnd(LinkedListNode head, int n) {
        LinkedListNode first = head, second = head;
        for (int i = 0; i < n; i++) {
            if (first == null) return null;
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        return second; // O(n)
    }

    // 1️⃣1️⃣ Merge Two Sorted Linked Lists
    public static LinkedListNode mergeSortedLists(LinkedListNode l1, LinkedListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.data < l2.data) {
            l1.next = mergeSortedLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeSortedLists(l1, l2.next);
            return l2;
        }
    }

    // 1️⃣2️⃣ Detect Intersection
    public static LinkedListNode getIntersectionNode(LinkedListNode headA, LinkedListNode headB) {
        Set<LinkedListNode> set = new HashSet<>();
        while (headA != null) {
            set.add(headA);
            headA = headA.next;
        }
        while (headB != null) {
            if (set.contains(headB)) return headB;
            headB = headB.next;
        }
        return null; // O(n), O(n)
    }

    // 1️⃣3️⃣ Circular Linked List Traversal
    public static void traverseCircularList(LinkedListNode head) {
        if (head == null) return;
        LinkedListNode curr = head;
        do {
            System.out.print(curr.data + " -> ");
            curr = curr.next;
        } while (curr != head);
        System.out.println("HEAD");
    }

    // 1️⃣4️⃣ Add Two Numbers Represented as Linked Lists
    public static LinkedListNode addTwoNumbers(LinkedListNode l1, LinkedListNode l2) {
        LinkedListNode dummy = new LinkedListNode(0);
        LinkedListNode curr = dummy;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int sum = carry;
            if (l1 != null) { sum += l1.data; l1 = l1.next; }
            if (l2 != null) { sum += l2.data; l2 = l2.next; }
            curr.next = new LinkedListNode(sum % 10);
            carry = sum / 10;
            curr = curr.next;
        }
        return dummy.next;
    }

    // 1️⃣5️⃣ Partition Linked List Around Value x
    public static LinkedListNode partition(LinkedListNode head, int x) {
        LinkedListNode before = new LinkedListNode(0), after = new LinkedListNode(0);
        LinkedListNode b = before, a = after;
        while (head != null) {
            if (head.data < x) { b.next = head; b = b.next; }
            else { a.next = head; a = a.next; }
            head = head.next;
        }
        a.next = null;
        b.next = after.next;
        return before.next; // O(n)
    }

    // 1️⃣6️⃣ Swap Nodes in Pairs
    public static LinkedListNode swapPairs(LinkedListNode head) {
        LinkedListNode dummy = new LinkedListNode(0);
        dummy.next = head;
        LinkedListNode curr = dummy;
        while (curr.next != null && curr.next.next != null) {
            LinkedListNode first = curr.next;
            LinkedListNode second = curr.next.next;
            first.next = second.next;
            second.next = first;
            curr.next = second;
            curr = first;
        }
        return dummy.next;
    }

    // 1️⃣7️⃣ Clone Linked List with Random Pointer
    public static LinkedListNode cloneRandomList(LinkedListNode head) {
        if (head == null) return null;
        LinkedListNode curr = head;
        while (curr != null) {
            LinkedListNode copy = new LinkedListNode(curr.data);
            copy.next = curr.next;
            curr.next = copy;
            curr = copy.next;
        }
        curr = head;
        while (curr != null) {
            if (curr.random != null) curr.next.random = curr.random.next;
            curr = curr.next.next;
        }
        LinkedListNode dummy = new LinkedListNode(0), copyCurr = dummy;
        curr = head;
        while (curr != null) {
            copyCurr.next = curr.next;
            copyCurr = copyCurr.next;
            curr.next = curr.next.next;
            curr = curr.next;
        }
        return dummy.next;
    }

    // 1️⃣8️⃣ Segregate Even and Odd Nodes
    public static LinkedListNode segregateEvenOdd(LinkedListNode head) {
        LinkedListNode evenStart = null, evenEnd = null, oddStart = null, oddEnd = null;
        while (head != null) {
            if (head.data % 2 == 0) {
                if (evenStart == null) evenStart = evenEnd = head;
                else { evenEnd.next = head; evenEnd = evenEnd.next; }
            } else {
                if (oddStart == null) oddStart = oddEnd = head;
                else { oddEnd.next = head; oddEnd = oddEnd.next; }
            }
            head = head.next;
        }
        if (evenStart == null) return oddStart;
        evenEnd.next = oddStart;
        if (oddEnd != null) oddEnd.next = null;
        return evenStart;
    }

    // 1️⃣9️⃣ Flatten Multilevel Linked List
    public static LinkedListNode flatten(LinkedListNode head) {
        if (head == null) return null;
        LinkedListNode curr = head;
        while (curr != null) {
            if (curr.random != null) { // using random as child pointer
                LinkedListNode temp = curr.next;
                curr.next = flatten(curr.random);
                LinkedListNode tail = curr.next;
                while (tail.next != null) tail = tail.next;
                tail.next = temp;
                curr.random = null;
            }
            curr = curr.next;
        }
        return head;
    }

    // 2️⃣0️⃣ Find Intersection Node in Y-shaped Linked List
    public static LinkedListNode findIntersectionY(LinkedListNode head1, LinkedListNode head2) {
        int len1 = length(head1), len2 = length(head2);
        while (len1 > len2) { head1 = head1.next; len1--; }
        while (len2 > len1) { head2 = head2.next; len2--; }
        while (head1 != null && head2 != null) {
            if (head1 == head2) return head1;
            head1 = head1.next;
            head2 = head2.next;
        }
        return null;
    }

    private static int length(LinkedListNode head) {
        int len = 0;
        while (head != null) { head = head.next; len++; }
        return len;
    }

    // Main method (Optional test)
    public static void main(String[] args) {
        LinkedListNode head = null;
        head = insertAtTail(head, 1);
        head = insertAtTail(head, 2);
        head = insertAtTail(head, 3);
        head = insertAtTail(head, 4);
        System.out.println("Original List:");
        LinkedListNode curr = head;
        while (curr != null) {
            System.out.print(curr.data + " -> ");
            curr = curr.next;
        }
        System.out.println("NULL");
    }
}
