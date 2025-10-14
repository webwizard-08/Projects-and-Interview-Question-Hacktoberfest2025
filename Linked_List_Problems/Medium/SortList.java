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
    public ListNode sortList(ListNode head) {

        if(head == null || head.next == null){
            return head;
        }

        ListNode slow = head;
        ListNode fast = head;
        ListNode temp = head;

        while(fast != null && fast.next != null){
            temp = slow;
            slow = slow.next;
            fast = fast.next.next;
        }

        temp.next = null;

        ListNode left = sortList(head);
        ListNode right = sortList(slow);

        return merge(left, right);
        
    }

    public ListNode merge(ListNode left, ListNode right){
        ListNode ans = new ListNode(0);
        ListNode cur = ans;

        while(left != null && right != null){
            if(left.val < right.val){
                cur.next = left;
                left = left.next;
            }
            else{
                cur.next = right;
                right = right.next;
            }

            cur = cur.next;
        }


        if(left != null){
            cur.next = left;
            left = left.next;
        }

        if(right != null){
            cur.next = right;
            right = right.next;
        }
        return ans.next; 
    }
}