package com.jml.classifiers;

import java.util.ArrayList;
import java.util.List;

/**
 * A k-dimensional tree or K-d tree is a binary tree which partitions a space by organizing points in a K-dimensional
 * space. K-d trees can be used to solve generalized N-point problems e.g. K-nearest neighbors.
 */
public class KDTree {

    /**
     * Dimension of each point in the tree.
     */
    private final int k;
    private PointNode root;

    /**
     *
     * @param k Dimension of each point in the k dimensional tree.
     */
    public KDTree(int k) {
        if(k<1) {
            throw new IllegalArgumentException("k must be positive integer but got k=" + k + ".");
        }

        this.k = k;
    }


    /**
     * Inserts a point into this K-d tree.
     * @param point Point to insert.
     */
    public void insert(double[] point) {

        if(point.length!=k) {
            throw new IllegalArgumentException("Point does not have the same number of dimensions as the other points" +
                    "in the tree. Expecting point with dimension " + k + " but got " + point.length);
        }

        if(root==null) { // Then create a root.
            this.root = new PointNode(point);
        } else {
            PointNode current = root;
            int depth = 0;
            int axis;

            while(current!=null) {
                axis = depth % k;

                if(point[axis] < current.getValue(axis) ) {
                    if(current.left == null) {
                        current.left = new PointNode(point);
                        break;
                    } else {
                        current = current.left;
                    }
                } else {
                    if(current.right == null) {
                        current.right = new PointNode(point);
                        break;
                    } else {
                        current = current.right;
                    }
                }

                depth++;
            }
        }
    }



    /**
     * Traverses this K-d tree in order.
     * @return Returns an arraylist containing the points from the in-order traversal of the tree.
     */
    public List<double[]> inOrder() {
        List<double[]> pointList = new ArrayList<>();
        return inOrder(root, pointList);
    }


    /**
     * Traverses this K-d tree in order.
     *
     * @param current Current Node in the traversal.
     * @param pointList Current list of points visited in the traversal.
     * @return Returns the points in order as a List.
     */
    private List<double[]> inOrder(PointNode current, List<double[]> pointList) {
        if(current!=null) {
            inOrder(current.left, pointList);
            pointList.add(current.get());
            inOrder(current.right, pointList);
        }

        return pointList;
    }


    /**
     * Node which contains a k-dimensional point.
     */
    class PointNode {
        double[] point;
        PointNode left, right;


        /**
         * Creates a PointNode with a specified point.
         * @param point Point to insert into this PointNode.
         */
        public PointNode(double[] point) {
            if(point.length!=k) {
                throw new IllegalArgumentException("Point does not have the same number of dimensions as the other points" +
                        "in the tree. Expecting point with dimension " + k + " but got " + point.length);
            }

            this.point = point;
        }


        /**
         * Gets the point from the PointNode
         * @return Returns the point from the PointNode.
         */
        public double[] get() {
           return this.point;
        }


        /**
         * Gets the value of a point at the specified dimension.
         * @param dimension Dimension of point to get value of.
         *
         * @return
         */
        double getValue(int dimension) {
            return point[dimension];
        }
    }
}
