/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        int count = 1;
        ListNode temp = head;
        if(head == null)return null;

        while(temp.next != null){
            temp = temp.next;
            count++;
        }

        ListNode dummy = head;
        k = k % count;
        if(k == 0)return head;

        for(int i=0; i<count-k-1; i++){
            dummy = dummy.next;
        }

        temp.next = head;
        head = dummy.next;
        dummy.next = null;
        return head;
    }
}