#!/usr/bin/env python3

def max_depth_below(self):
    """
    Returns the maximum depth below this node (including leaves)
    """
    # If node is a leaf (safety check)
    if self.is_leaf:
        return self.depth

    left_depth = self.left_child.max_depth_below() if self.left_child else self.depth
    right_depth = self.right_child.max_depth_below() if self.right_child else self.depth

    return max(left_depth, right_depth)
