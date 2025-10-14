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
    public void reorderList(ListNode head) {
        ListNode p = head;
        ListNode q = null;
        ListNode temp = head;

        if(temp == null || temp.next == null){
            return;
        }

        while(p.next != null){
            q = p;
            p = p.next;
        }

        q.next = null;
        p.next = temp.next;
        temp.next = p;

        temp = p.next;

        reorderList(temp);
    }
}