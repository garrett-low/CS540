/*
 * This class represents a node in a decision tree. 
 * 
 * It has fields for:
 * - feature - the feature on which split at this node happens
 * - threshold - the value of the threshold for the feature chosen
 * - left - pointer to the left child node
 * - right - pointer to the right child node 
 * - classLabel - the value 2 or 4 which represents label assigned 
 * 			to instances at this node if this is a leaf node
 */


public class DecTreeNode {
	
	public int feature;
	public int threshold;
	public DecTreeNode left = null;
	public DecTreeNode right = null; 
	public int classLabel; 
	
	public DecTreeNode(int classLabel, int feature, int threshold) {
		
		this.classLabel = classLabel;
		this.feature = feature;
		this.threshold = threshold;
		
	}
	
	public boolean isLeaf() {
		return this.left == null && this.right == null;
	}
}
