import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class P2 {

    // This code uses all 9 features to train the decision tree
    public static int numAttr = 9;
    private static final String outputPath = "./output.txt";
    private static final String trainPath = "./breast-cancer-wisconsin.data";
    private static final String testPath = "./test_set_grlow.txt";
    private static final DecimalFormat fourPlaces = new DecimalFormat("0.0000");
    private static DecTreeNode rootMaxDepth;

    public static void main(String[] args) {
        createFile(outputPath);

        try (FileWriter writer = new FileWriter(outputPath)) {
            List<ArrayList<Integer>> trainData = cleanAndTransformData(trainPath); // Import training set

            q1ImportDataCheck(trainData, writer); // Q1 - count imported data
            q2EntropyAtRoot(trainData, writer); // Q2 - entropy at root
            q3Stump(trainData, 6, writer); // Q3 - decision stump: feature 8 >> index 6

            // Build training tree
//            DecTreeNode root = buildTree(trainData);
//            printTree(root, writer);
//            printSideways(root, "");

            // implement method for choosing specific features out of all features of train dataset
            // feature 4, 7, 8, 9, 10 >> index 2, 5, 6, 7, 8
            int[] featureArray = {2, 5, 6, 7, 8};
            DecTreeNode rootWithFeature = buildTreeWithFeature(trainData, featureArray);
            printSideways(rootWithFeature, "");

            printTree(rootWithFeature, writer); // Q5 - print the training tree with the given features

            // implement method for finding maximum depth of the tree
            // Q6 - depth of training tree
            int depthRootWithFeature = getDepth(rootWithFeature);
            System.out.println("Depth of training tree with features: " + depthRootWithFeature);
            writer.write("Depth of training tree with features: " + depthRootWithFeature + "\n");

            // implement method for classifying test data given the root of the decision tree
            // Import test data
            List<ArrayList<Integer>> testData = cleanAndTransformData(testPath);
            List<ArrayList<Integer>> labelledTestData = classifyData(rootWithFeature, testData);

            System.out.println("Labelled patient data:");
            writer.write("Labelled patient data:\n");
            for (int i = 0; i < labelledTestData.size(); i++) {
                System.out.print(labelledTestData.get(i).get(9));
                writer.write(String.valueOf(labelledTestData.get(i).get(9)));
                if (i < labelledTestData.size() - 1) {
                    System.out.print(",");
                    writer.write(",");
                }
            }
            System.out.println();
            writer.write("\n");

            // TODO: need to implement method for pruning the tree to have a fixed maximum depth
            rootMaxDepth = buildTreeWithFeatureAndDepth(trainData, featureArray, 7, 0);
            System.out.println("Depth of 'pruned' tree: " + getDepth(rootMaxDepth));
            writer.write("Depth of 'pruned' tree: " + getDepth(rootMaxDepth) + "\n");
            printTree(rootMaxDepth, writer);
            printSideways(rootMaxDepth, "");

            List<ArrayList<Integer>> labelledPrunedTestData = classifyData(rootMaxDepth, testData);

            System.out.println("Pruned and labelled patient data:");
            writer.write("Pruned and labelled patient data:\n");
            for (int i = 0; i < labelledPrunedTestData.size(); i++) {
                System.out.print(labelledPrunedTestData.get(i).get(9));
                writer.write(String.valueOf(labelledPrunedTestData.get(i).get(9)));
                if (i < labelledPrunedTestData.size() - 1) {
                    System.out.print(",");
                    writer.write(",");
                }
            }
            System.out.println();
            writer.write("\n");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void q1ImportDataCheck(List<ArrayList<Integer>> trainData, FileWriter writer) throws IOException {
        int benignCount = 0;
        int malignantCount = 0;
        for (ArrayList<Integer> patientData : trainData) {
            Integer classLabel = patientData.get(9);
            if (classLabel == 2) {
                benignCount++;
            } else {
                malignantCount++;
            }
        }
        writer.write("Q1 - benign, malignant: " + benignCount + "," + malignantCount + "\n");
        System.out.println("Q1 - benign, malignant: " + benignCount + "," + malignantCount);
    }

    /*
     * This method cleans the train dataset, removes rows with missing values,
     * and ignores first id column, since that's not used during training.
     */
    public static List<ArrayList<Integer>> cleanAndTransformData(String file) {

        List<ArrayList<Integer>> cleanData = new ArrayList<ArrayList<Integer>>();

        try {
            Scanner scan = new Scanner(new File(file));

            while (scan.hasNextLine()) {
                String line = scan.nextLine();
                // if the line doesn't contain '?', we should include it into our clean dataset
                if (line.indexOf('?') == -1) {
                    String[] split = line.split(",");

                    ArrayList<Integer> instance = new ArrayList<Integer>();
                    // ignore id column, it's not used for training
                    // convert string values to integer values
                    for (int i = 1; i < split.length; i++) {
                        instance.add(Integer.parseInt(split[i]));
                    }
                    cleanData.add(instance);
                }
            }
            scan.close();
        } catch (FileNotFoundException e) {
            System.out.println("File with the name " + file + " cannot be read!");
        }
        return cleanData;
    }

    /*
     * Method for computing entropy.
     */
    public static double entropy(double p0) {
        if (p0 == 0 || p0 == 1) return 0;

        double p1 = 1 - p0;
        return -(p0 * Math.log(p0) / Math.log(2) + p1 * Math.log(p1) / Math.log(2));
    }

    /*
     * Method for computing information gain.
     */
    public static double informationGain(List<ArrayList<Integer>> dataSet, int feature, int threshold) {

        int dataSize = dataSet.size();
        // first computing H(Y)
        int count = 0;
        for (List<Integer> data : dataSet) {
            if (data.get(data.size() - 1) == 2) { // if label equals 2, last column is label
                count++;
            }
        }
        double Hy = entropy((double) count / dataSize);

        // computing H(Y|X), conditional entropy
        double Hyx = 0;
        int countLess = 0;
        int countGreater = 0;
        int countLessAndPositive = 0;
        int countGreaterAndPositive = 0;
        for (List<Integer> data : dataSet) {
            if (data.get(feature) <= threshold) {
                countLess++;
                if (data.get(data.size() - 1) == 2) countLessAndPositive++; // last column is label
            } else {
                countGreater++;
                if (data.get(data.size() - 1) == 2) countGreaterAndPositive++;
            }
        }
        double prob1 = (double) countLess / dataSize;
        double prob2 = (double) countGreater / dataSize;
        if (prob1 > 0) {
            Hyx = Hyx + prob1 * entropy(((double) countLessAndPositive) / countLess);
        }
        if (prob2 > 0) {
            Hyx = Hyx + prob2 * entropy(((double) countGreaterAndPositive) / countGreater);
        }
        // return difference between entropy and conditional entropy
        return Hy - Hyx;
    }

    /*
     * Method for building a decision tree using the given dataset.
     * It returns a pointer to the root node.
     */
    private static DecTreeNode buildTree(List<ArrayList<Integer>> dataSet) {
        int numData = dataSet.size();
        int bestAttr = -1;
        int bestThreshold = Integer.MIN_VALUE;
        double bestScore = Double.NEGATIVE_INFINITY;
        boolean leaf = false;
        DecTreeNode node = null;

        // if node isn't leaf node, compute the best split for it
        if (!leaf) {
            for (int j = 0; j < numAttr; j++) {
                for (int i = 1; i < 11; i++) {
                    double score = informationGain(dataSet, j, i);
                    if (score > bestScore) {
                        bestScore = score;
                        bestAttr = j;
                        bestThreshold = i;
                    }
                }
            }
            if (bestScore == 0) {
                leaf = true;
            }

            // split the entire # of instances into two groups based on the threshold value (<= and >)
            List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
            List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
            for (ArrayList<Integer> data : dataSet) {
                if (data.get(bestAttr) <= bestThreshold) {
                    leftList.add(data);
                } else {
                    rightList.add(data);
                }
            }

            if (leftList.size() == 0 || rightList.size() == 0) {
                leaf = true;
            }
            // if node is not leaf, create left and right children
            if (!leaf) {
                node = new DecTreeNode(-1, bestAttr, bestThreshold);
                node.left = buildTree(leftList);
                node.right = buildTree(rightList);
            }
        }
        // if node is leaf, need to count # of instances with labels 2 and 4
        if (leaf) {
            int count = 0;
            for (List<Integer> data : dataSet) {
                if (data.get(data.size() - 1) == 2)
                    count += 1;
            }

            if (count >= numData - count) {
                node = new DecTreeNode(2, -1, -1); // assign label 2 to the leaf node
            } else {
                node = new DecTreeNode(4, -1, -1); // assign label 4 to the leaf node
            }
        }
        return node;
    }

    private static void q2EntropyAtRoot(List<ArrayList<Integer>> trainData, FileWriter writer) throws IOException {
        int dataSize = trainData.size();
        // first computing H(Y)
        int count = 0;
        for (List<Integer> data : trainData) {
            if (data.get(data.size() - 1) == 2) { // if label equals 2, last column is label
                count++;
            }
        }
        double Hy = entropy((double) count / dataSize);
        writer.write("Q2 - entropy at root before split: " + fourPlaces.format(Hy) + "\n");
        System.out.println("Q2 - entropy at root before split: " + fourPlaces.format(Hy));
    }

    /*
     * Method for building a decision tree using the given dataset and given features.
     * It returns a pointer to the root node.
     */
    private static DecTreeNode buildTreeWithFeature(List<ArrayList<Integer>> dataSet, int[] featureArray) {
        int numData = dataSet.size();
        int bestAttr = -1;
        int bestThreshold = Integer.MIN_VALUE;
        double bestScore = Double.NEGATIVE_INFINITY;
        boolean leaf = false;
        DecTreeNode node = null;

        // if node isn't leaf node, compute the best split for it
        if (!leaf) {
            for (int feature : featureArray) {
                for (int i = 1; i < 11; i++) {
                    double score = informationGain(dataSet, feature, i);
                    if (score > bestScore) {
                        bestScore = score;
                        bestAttr = feature;
                        bestThreshold = i;
                    }
                }
            }
            if (bestScore == 0) {
                leaf = true;
            }

            // split the entire # of instances into two groups based on the threshold value (<= and >)
            List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
            List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
            for (ArrayList<Integer> data : dataSet) {
                if (data.get(bestAttr) <= bestThreshold) {
                    leftList.add(data);
                } else {
                    rightList.add(data);
                }
            }

            if (leftList.size() == 0 || rightList.size() == 0) {
                leaf = true;
            }
            // if node is not leaf, create left and right children
            if (!leaf) {
                node = new DecTreeNode(-1, bestAttr, bestThreshold);
                node.left = buildTreeWithFeature(leftList, featureArray);
                node.right = buildTreeWithFeature(rightList, featureArray);
            }
        }
        // if node is leaf, need to count # of instances with labels 2 and 4
        if (leaf) {
            int count = 0;
            for (List<Integer> data : dataSet) {
                if (data.get(data.size() - 1) == 2)
                    count += 1;
            }

            if (count >= numData - count) {
                node = new DecTreeNode(2, -1, -1); // assign label 2 to the leaf node
            } else {
                node = new DecTreeNode(4, -1, -1); // assign label 4 to the leaf node
            }
        }
        return node;
    }

    /*
     * Method for building a decision tree using the given dataset and given features.
     * It returns a pointer to the root node.
     */
    private static DecTreeNode buildTreeWithFeatureAndDepth(List<ArrayList<Integer>> dataSet, int[] featureArray, int depth, int currentDepth) {
        int numData = dataSet.size();
        int bestAttr = -1;
        int bestThreshold = Integer.MIN_VALUE;
        double bestScore = Double.NEGATIVE_INFINITY;
        boolean leaf = false;
        DecTreeNode node = null;

        // if node isn't leaf node, compute the best split for it
        if (!leaf) {
            for (int feature : featureArray) {
                for (int i = 1; i < 11; i++) {
                    double score = informationGain(dataSet, feature, i);
                    if (score > bestScore) {
                        bestScore = score;
                        bestAttr = feature;
                        bestThreshold = i;
                    }
                }
            }
            if (bestScore == 0) {
                leaf = true;
            }

            // split the entire # of instances into two groups based on the threshold value (<= and >)
            List<ArrayList<Integer>> leftList = new ArrayList<>();
            List<ArrayList<Integer>> rightList = new ArrayList<>();
            if (currentDepth < depth) {
                for (ArrayList<Integer> data : dataSet) {
                    if (data.get(bestAttr) <= bestThreshold) {
                        leftList.add(data);
                    } else {
                        rightList.add(data);
                    }
                }
            }

            if (leftList.size() == 0 || rightList.size() == 0) {
                leaf = true;
            }
            // if node is not leaf, create left and right children
            if (!leaf && currentDepth < depth) {
                node = new DecTreeNode(-1, bestAttr, bestThreshold);
                currentDepth++;
                node.left = buildTreeWithFeatureAndDepth(leftList, featureArray, 7, currentDepth);
                node.right = buildTreeWithFeatureAndDepth(rightList, featureArray, 7, currentDepth);
            }
        }

        // if node is leaf, need to count # of instances with labels 2 and 4
        if (leaf) {
            int count = 0;
            for (List<Integer> data : dataSet) {
                if (data.get(data.size() - 1) == 2)
                    count += 1;
            }

            if (count >= numData - count) {
                node = new DecTreeNode(2, -1, -1); // assign label 2 to the leaf node
            } else {
                node = new DecTreeNode(4, -1, -1); // assign label 4 to the leaf node
            }
        }
        return node;
    }

    /*
     * Method for building a decision tree using the given dataset.
     * It returns a pointer to the root node.
     */
    private static void q3Stump(List<ArrayList<Integer>> dataSet, int featureStump, FileWriter writer) throws IOException {
        int bestThreshold = Integer.MIN_VALUE;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int i = 1; i < 11; i++) {
            double score = informationGain(dataSet, featureStump, i);
            if (score > bestScore) {
                bestScore = score;
                bestThreshold = i;
            }
        }

        // split the entire # of instances into two groups based on the threshold value (<= and >)
        List<ArrayList<Integer>> leftList = new ArrayList<ArrayList<Integer>>();
        List<ArrayList<Integer>> rightList = new ArrayList<ArrayList<Integer>>();
        for (ArrayList<Integer> data : dataSet) {
            if (data.get(featureStump) <= bestThreshold) {
                leftList.add(data);
            } else {
                rightList.add(data);
            }
        }

        int leftBenignCount = 0, leftMalignantCount = 0;
        for (ArrayList<Integer> patient : leftList) {
            if (patient.get(patient.size() - 1) == 2) {
                leftBenignCount++;
            } else {
                leftMalignantCount++;
            }
        }

        int rightBenignCount = 0, rightMalignantCount = 0;
        for (ArrayList<Integer> patient : rightList) {
            if (patient.get(patient.size() - 1) == 2) {
                rightBenignCount++;
            } else {
                rightMalignantCount++;
            }
        }

        int[] q3array = {leftBenignCount, rightBenignCount, leftMalignantCount, rightMalignantCount};

        writer.write("Q3 - stump positive and negative counts (above-benign, below-benign, above-malignant, below-malignant): " + Arrays.toString(q3array) + "\n");
        System.out.println("Q3 - stump positive and negative counts (above-benign, below-benign, above-malignant, below-malignant): " + Arrays.toString(q3array));

        writer.write("Q4 - information gain: " + fourPlaces.format(bestScore) + "\n");
        System.out.println("Q4 - information gain: " + fourPlaces.format(bestScore));
    }

    // TODO: need to implement method for printing the tree in the correct format
    /*
     * Method which given the root of the Decision Tree, prints it
     * in the format specified at the webpage for P2 assignment.
     */
    private static void printTree(DecTreeNode node, FileWriter writer) throws IOException {
        System.out.println(preOrderTraversal(node, "", "", writer));
        writer.write("\n");
    }

    private static String preOrderTraversal(DecTreeNode node, String indent, String prefix, FileWriter writer) throws IOException {
        String preOrderString = "";

        if (node != null) {
            String feature = String.valueOf(node.feature + 2);
            String threshold = String.valueOf(node.threshold);
            String classLabel = String.valueOf(node.classLabel);

            System.out.print(prefix);
            writer.write(prefix);
            if (node.isLeaf()) {
                System.out.print(" return " + classLabel);
                writer.write(" return " + classLabel);
            }

            String leftPrefix = "\n" + indent + "if (x" + feature + " <= " + threshold + ")";
            String rightPrefix = "\n" + indent + "else";

            preOrderTraversal(node.left, indent + " ", leftPrefix, writer);
            preOrderTraversal(node.right, indent + " ", rightPrefix, writer);
        }

        return preOrderString;
    }

    private static int getDepth(DecTreeNode node) {
        if (node == null || node.isLeaf()) {
            return 0;
        }

        int leftDepth = getDepth(node.left);
        int rightDepth = getDepth(node.right);
        return 1 + Math.max(leftDepth, rightDepth);
    }

    private static List<ArrayList<Integer>> classifyData(DecTreeNode node, List<ArrayList<Integer>> data) {
        List<ArrayList<Integer>> labelledTestData = new ArrayList<>();

        for (ArrayList<Integer> patient : data) {
            patient.add(classifyRecursive(node, patient));
            labelledTestData.add(patient);
        }

        return labelledTestData;
    }

    private static Integer classifyRecursive(DecTreeNode node, ArrayList<Integer> patient) {
        int label = -1;

        if (node == null) {
            return label;
        }

        if (node.isLeaf()) {
            return node.classLabel;
        } else if (patient.get(node.feature) <= node.threshold) {
            label = classifyRecursive(node.left, patient);
        } else {
            label = classifyRecursive(node.right, patient);
        }

        return label;
    }

    private static void printSideways(DecTreeNode current, String indent) {
        if (current != null) {
            printSideways(current.right, indent + "    ");
            System.out.println(indent + "[feat:" + current.feature + ", thresh:" + current.threshold + ", label:" + current.classLabel + "]");
            printSideways(current.left, indent + "    ");
        }
    }

    private static void createFile(String outputPath) {
        try {
            File myObj = new File(outputPath);
            if (myObj.createNewFile()) {
                System.out.println("File created: " + myObj.getName());
            } else {
                System.out.println("File already exists.");
            }
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}