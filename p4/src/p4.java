import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class p4 {

    private static final int K = 7;               // number of clusters to generate
    private static final int FEATURE_LENGTH = 3;  // number of parameters used in the algorithms

    // inputs and outputs
    private static final String covidDataPath = "./time_series_covid19_deaths_global.csv";
    private static final Date date = Calendar.getInstance().getTime();
    private static final DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd-HH.mm.ss.SSS");
    private static final String strDate = dateFormat.format(date);
    private static final String outputPath = "./temp/output-" + strDate + ".txt";
    private static List<Double[]> finalMeans = new ArrayList<>();
    private static DecimalFormat fourPlaces = new DecimalFormat("0.0000");

    public static void main(String[] args) {
        createFile(outputPath);

        try (FileWriter writer = new FileWriter(outputPath)) {
            /*
             * For your submission you need to implement Part 1 for finding suitable parameters.
             */
            // TODO: remove TA stuff
            Map<String, Double[]> globalData = preprocessData(covidDataPath);

            // Q1 - per country time series
            Map<String, Integer[]> globalTimeSeries = q1SumCountryTimeSeries(covidDataPath);

            Set<String> countries = globalTimeSeries.keySet();
            List<String> sortedListCountries = new ArrayList<>(countries);
            Collections.sort(sortedListCountries);

            writer.write("==========================================================================\n");
            writer.write("Q1 - Per Country time series:\n");
            writer.write("==========================================================================\n");
            q1PrintIntegerArrayFromMap(writer, globalTimeSeries, sortedListCountries, true);

            // Q2 - Differenced time series
            Map<String, Integer[]> globalDiffTimeSeries = q2CalcDiffTimeSeries(globalTimeSeries);
            writer.write("==========================================================================\n");
            writer.write("Q2 - Per Country differenced time series:\n");
            writer.write("==========================================================================\n");
            q1PrintIntegerArrayFromMap(writer, globalDiffTimeSeries, sortedListCountries, true);

            // Q4 - Hundred day plot parameters
            Map<String, Integer[]> globalDaysToDouble = q4CountDaysToDouble(globalTimeSeries);
            writer.write("==========================================================================\n");
            writer.write("Q4 - Countries and days to double:\n");
            writer.write("==========================================================================\n");
            Set<String> affectedCountries = globalDaysToDouble.keySet();
            List<String> sortedAffectedCountries = new ArrayList<>(affectedCountries);
            Collections.sort(sortedAffectedCountries);
            q1PrintIntegerArrayFromMap(writer, globalDaysToDouble, sortedAffectedCountries, true);
            writer.write("==========================================================================\n");
            writer.write("Q4 output, no country label:\n");
            writer.write("==========================================================================\n");
            q1PrintIntegerArrayFromMap(writer, globalDaysToDouble, sortedAffectedCountries, false);

            // Q5 - Hierarchical clustering, single-linkage, euclidean
            Map<String, Integer> clusterHierarchicalSingle = q5CalcHierarchicalSingle(writer, globalDaysToDouble, sortedAffectedCountries);
            writer.write("==========================================================================\n");
            writer.write("Q5 - output, hierarchical, single, euclidean, no country label:\n");
            writer.write("==========================================================================\n");
            q5PrintCluster(writer, clusterHierarchicalSingle, sortedAffectedCountries);

            // Q6 - Hierarchical clustering, complete linkage, euclidean
            Map<String, Integer> clusterHierarchicalComplete = q6CalcHierarchicalComplete(writer, globalDaysToDouble, sortedAffectedCountries);
            writer.write("==========================================================================\n");
            writer.write("Q6 - output, hierarchical, complete, euclidean, no country label:\n");
            writer.write("==========================================================================\n");
            q5PrintCluster(writer, clusterHierarchicalComplete, sortedAffectedCountries);

            // Q7 - K Means clustering, euclidean
            writer.write("==========================================================================\n");
            writer.write("Q7 - human-readable K clusters:\n");
            List<Double[]> globalDaysToDoubleList = new ArrayList<>();
            for (String country : sortedAffectedCountries) {
                Integer[] integerList = globalDaysToDouble.get(country);
                Double[] doubleList = new Double[integerList.length];
                for (int i = 0; i < integerList.length; i++) {
                    doubleList[i] = Double.valueOf(integerList[i]);
                }
                globalDaysToDoubleList.add(doubleList);
            }

            Map<String, Integer> kClusters = q7KMeans(writer, sortedAffectedCountries, globalDaysToDoubleList);
            writer.write("==========================================================================\n");
            writer.write("Q7 - output, K-means, euclidean:\n");
            writer.write("==========================================================================\n");
            q5PrintCluster(writer, kClusters, sortedAffectedCountries);
            writer.write("==========================================================================\n");
            writer.write("Q8 - output, K-means cluster centers:\n");
            writer.write("==========================================================================\n");
            q7PrintClusterCenters(writer, finalMeans);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static void q7PrintClusterCenters(FileWriter writer, List<Double[]> finalMeans) throws IOException {
        for (int i = 0; i < finalMeans.size(); i++) {
            Double[] vector = finalMeans.get(i);
            for (int j = 0; j < vector.length; j++) {
                writer.write(fourPlaces.format(vector[j]));
                if (j < vector.length - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n");
        }
        writer.write("\n");
    }

    private static void taSolutionKMeans(List<String> sortedListCountries, List<Double[]> global_data) {
        int[] cluster_info = new int[sortedListCountries.size()]; // cluster to which country is assigned to

        // K vectors of length FEATURE_LENGTH, those will hold our cluster centers
        List<Double[]> means = new ArrayList<Double[]>();
        // Choose K random countries as our initial means for cluster centers
        for (int i = 0; i < K; i++) {
            Random r = new Random();
            int rand = r.nextInt(sortedListCountries.size());
            Double[] vector = global_data.get(rand);
            means.add(vector);
        }

        // find initial cluster assignment for all countries
        findClusterForCountries(global_data, means, cluster_info);

        // start updating means for clusters until clusters do not change
        boolean keepUpdatingMeans = true;

        List<Double[]> previous_iteration_means = means;

        while (keepUpdatingMeans) {

            List<Double[]> recomputed_means = recomputeMeans(global_data, cluster_info);

            // need to make sure that all K means are not changing for stopping learning algorithm
            int checker = 0;
            for (int cl_index = 0; cl_index < K; cl_index++) {

                if (Arrays.equals(recomputed_means.get(cl_index), previous_iteration_means.get(cl_index))) {
                    checker += 1;
                }
            }
            if (checker == K) {
                // all K vectors are still the same after last iteration of updates to the means
                // therefore stop the algorithm
                keepUpdatingMeans = false;
                break;
            }
            findClusterForCountries(global_data, recomputed_means, cluster_info);
            previous_iteration_means = recomputed_means;
        }

        // Print out list of countries in each cluster for K-Means
        System.out.println("K-Means Clustering Results: ");
        for (int i = 1; i < K + 1; i++) {
            System.out.print(i - 1 + " [");
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < sortedListCountries.size(); j++) {
                if (cluster_info[j] == i) {
                    sb.append(sortedListCountries.get(j) + ", ");
                }
            }
            if (sb.length() != 0) System.out.print(sb.toString().substring(0, sb.length() - 2));
            else System.out.print(sb.toString());
            System.out.print("]");
            System.out.println();
        }
    }

    private static void q5PrintCluster(FileWriter writer, Map<String, Integer> cluster, List<String> sortedListCountries) throws IOException {
        boolean first = true;
        for (String country : sortedListCountries) {
            if (first) {
                first = false;
            } else {
                writer.write(",");
            }
            writer.write(String.valueOf(cluster.get(country)));
        }
        writer.write("\n");
    }

    private static Map<String, Integer> q5CalcHierarchicalSingle(FileWriter writer, Map<String, Integer[]> globalDaysToDouble,
                                                                 List<String> sortedAffectedCountries) throws IOException {
        List<List<String>> clusters = new ArrayList<>();
        HashMap<String, Integer> clusterHier = new HashMap<>();
        writer.write("==========================================================================\n");
        writer.write("Q5 - Hierarchical clustering, single linkage, Euclidean:\n");

        // each country appears as a separate cluster at the beginning
        for (int i = 0; i < sortedAffectedCountries.size(); i++) {
            List<String> cluster = new ArrayList<>();
            cluster.add(sortedAffectedCountries.get(i));
            clusters.add(cluster);
        }

        while (clusters.size() != K) {

            double minimumDistance = Double.MAX_VALUE;
            List<String> clusterToMerge1 = null, clusterToMerge2 = null;
            int indexCluster1 = -1, indexCluster2 = -1;

            for (int i = 0; i < clusters.size(); i++) {
                List<String> cluster1 = clusters.get(i);
                for (int j = i + 1; j < clusters.size(); j++) { // make sure it's not the same cluster though, otherwise dist is 0.

                    List<String> cluster2 = clusters.get(j);
                    // compute single linkage distance between given two clusters
                    double currDist = q5SingleEuclideanDistance(globalDaysToDouble, cluster1, cluster2);

                    if (currDist < minimumDistance) {
                        minimumDistance = currDist;
                        clusterToMerge1 = cluster1;
                        clusterToMerge2 = cluster2;
                        indexCluster1 = i;
                        indexCluster2 = j;
                    }
                }
            }

            // need to merge those two clusters with min/max distance
            if (clusterToMerge1 != null && clusterToMerge2 != null) {

                clusters.remove(indexCluster2);
                clusters.remove(indexCluster1);

                clusterToMerge1.addAll(clusterToMerge2);
                clusters.add(clusterToMerge1);
            } else {
                writer.write("Something is incorrect, one cluster might be null.\n");
            }
        }

        // Print out final clusters
        int cluster_num = 0;
        for (List<String> l : clusters) {
            writer.write(cluster_num + " " + l + "\n");

            for (int i = 0; i < l.size(); i++) {
                String country = l.get(i);
                clusterHier.put(country, cluster_num);
            }
            cluster_num++;
        }
        writer.write("==========================================================================\n");

        return clusterHier;
    }

    private static Map<String, Integer> q6CalcHierarchicalComplete(FileWriter writer, Map<String, Integer[]> globalDaysToDouble,
                                                                   List<String> sortedAffectedCountries) throws IOException {
        List<List<String>> clusters = new ArrayList<>();
        HashMap<String, Integer> clusterHier = new HashMap<>();

        writer.write("==========================================================================\n");
        writer.write("Q5 - Hierarchical clustering, single linkage, Euclidean:\n");

        // each country appears as a separate cluster at the beginning
        for (int i = 0; i < sortedAffectedCountries.size(); i++) {
            List<String> cluster = new ArrayList<>();
            cluster.add(sortedAffectedCountries.get(i));
            clusters.add(cluster);
        }

        while (clusters.size() != K) {

            double minOfMaxDistance = Double.MAX_VALUE;
            List<String> clusterToMerge1 = null, clusterToMerge2 = null;
            int indexCluster1 = -1, indexCluster2 = -1;

            for (int i = 0; i < clusters.size(); i++) {
                List<String> cluster1 = clusters.get(i);
                for (int j = i + 1; j < clusters.size(); j++) { // make sure it's not the same cluster though, otherwise dist is 0.

                    List<String> cluster2 = clusters.get(j);
                    // compute complete linkage distance between given two clusters
                    double currDist = q6CompleteEuclideanDistance(globalDaysToDouble, cluster1, cluster2);

                    if (currDist < minOfMaxDistance) {
                        minOfMaxDistance = currDist;
                        clusterToMerge1 = cluster1;
                        clusterToMerge2 = cluster2;
                        indexCluster1 = i;
                        indexCluster2 = j;
                    }
                }
            }

            // need to merge those two clusters with min/max distance
            if (clusterToMerge1 != null && clusterToMerge2 != null) {

                clusters.remove(indexCluster2);
                clusters.remove(indexCluster1);

                clusterToMerge1.addAll(clusterToMerge2);
                clusters.add(clusterToMerge1);
            } else {
                writer.write("Something is incorrect, one cluster might be null.\n");
            }
        }

        // Print out final clusters
        int cluster_num = 0;
        for (List<String> l : clusters) {
            writer.write(cluster_num + " " + l + "\n");

            for (int i = 0; i < l.size(); i++) {
                String country = l.get(i);
                clusterHier.put(country, cluster_num);
            }
            cluster_num++;
        }
        writer.write("==========================================================================\n");

        return clusterHier;
    }

    private static double q6CompleteEuclideanDistance(Map<String, Integer[]> globalData, List<String> cluster1, List<String> cluster2) {
        double maxDistance = -1.0;
        double distance;
        for (int i = 0; i < cluster1.size(); i++) {
            for (int j = 0; j < cluster2.size(); j++) {

                Integer[] values1 = globalData.get(cluster1.get(i));
                Integer[] values2 = globalData.get(cluster2.get(j));

                double x1 = values1[0];
                double y1 = values1[1];
                double z1 = values1[2];

                double x2 = values2[0];
                double y2 = values2[1];
                double z2 = values2[2];

                distance = q5CalcEuclidean(x1, y1, z1, x2, y2, z2);

                if (distance > maxDistance) maxDistance = distance;
            }
        }
        return maxDistance;
    }

    /*
     * This method computes single-linkage distance between two clusters.
     * Makes use of Euclidean distance.
     */
    private static double q5SingleEuclideanDistance(Map<String, Integer[]> globalData, List<String> cluster1, List<String> cluster2) {
        double minDistance = Double.MAX_VALUE;
        double distance;
        for (int i = 0; i < cluster1.size(); i++) {
            for (int j = 0; j < cluster2.size(); j++) {
                Integer[] values1 = globalData.get(cluster1.get(i));
                Integer[] values2 = globalData.get(cluster2.get(j));

                double x1 = values1[0];
                double y1 = values1[1];
                double z1 = values1[2];

                double x2 = values2[0];
                double y2 = values2[1];
                double z2 = values2[2];

                distance = q5CalcEuclidean(x1, y1, z1, x2, y2, z2);

                if (distance < minDistance) minDistance = distance;
            }
        }
        return minDistance;
    }

    private static double q5CalcEuclidean(double x1, double y1, double z1, double x2, double y2, double z2) {
        double distance;

        double xDiff = x1 - x2;
        double yDiff = y1 - y2;
        double zDiff = z1 - z2;

        distance = Math.sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);

        return distance;
    }

    private static void taSolutionHierchicalCluster(Map<String, Double[]> globalData, List<String> sortedListCountries) {
        List<List<String>> clusters = new ArrayList<>();

        // each country appears as a separate cluster at the beginning
        for (int i = 0; i < sortedListCountries.size(); i++) {
            List<String> cluster = new ArrayList<>();
            cluster.add(sortedListCountries.get(i));
            clusters.add(cluster);
        }

        while (clusters.size() != K) {

            double minimumDistance = Double.MAX_VALUE;
            List<String> clusterToMerge1 = null, clusterToMerge2 = null;
            int indexCluster1 = -1, indexCluster2 = -1;

            for (int i = 0; i < clusters.size(); i++) {
                List<String> cluster1 = clusters.get(i);
                for (int j = i + 1; j < clusters.size(); j++) { // make sure it's not the same cluster though, otherwise dist is 0.

                    List<String> cluster2 = clusters.get(j);
                    // compute single linkage distance between given two clusters
                    double currDist = singleLinkageDistance(globalData, cluster1, cluster2);

                    if (currDist < minimumDistance) {
                        minimumDistance = currDist;
                        clusterToMerge1 = cluster1;
                        clusterToMerge2 = cluster2;
                        indexCluster1 = i;
                        indexCluster2 = j;
                    }
                }
            }

            // need to merge those two clusters with min distance
            if (clusterToMerge1 != null && clusterToMerge2 != null) {

                clusters.remove(indexCluster2);
                clusters.remove(indexCluster1);

                clusterToMerge1.addAll(clusterToMerge2);
                clusters.add(clusterToMerge1);
            } else {
                System.out.println("Something is incorrect, one cluster might be null.");
            }
        }

        // Print out final clusters
        System.out.println("Hierarchical Clustering Results: ");
        int cluster_num = 0;
        HashMap<String, Integer> clusterHier = new HashMap<String, Integer>();
        for (List<String> l : clusters) {
            System.out.println(cluster_num + " " + l);

            for (int i = 0; i < l.size(); i++) {
                String country = l.get(i);
                clusterHier.put(country, cluster_num);
            }
            cluster_num++;
        }
        System.out.println("=============================================================");
    }

    private static Map<String, Integer[]> q4CountDaysToDouble(Map<String, Integer[]> globalTimeSeries) {
        Map<String, Integer[]> globalDaysToDouble = new HashMap<>();

        for (String country : globalTimeSeries.keySet()) {
            Integer[] timeSeries = globalTimeSeries.get(country);
            Integer nowDay = timeSeries.length - 1;
            Integer nowDeaths = timeSeries[nowDay];
            if (nowDeaths < 8) { // discard countries with very small deaths
                continue;
            }

            Integer halfDeaths = nowDeaths / 2;
            Integer quarterDeaths = nowDeaths / 4;
            Integer eighthDeaths = nowDeaths / 8;

            Integer halfDay = -1, quarterDay = -1, eighthDay = -1;
            Integer nowToHalf = -1, halfToQuarter = -1, quarterToEighth = -1;

            for (int i = timeSeries.length - 1; i >= 0; i--) {
                if (timeSeries[i] <= halfDeaths) {
                    halfDay = i;
                    break;
                }
            }

            for (int i = halfDay; i >= 0; i--) {
                if (timeSeries[i] <= quarterDeaths) {
                    quarterDay = i;
                    break;
                }
            }

            for (int i = quarterDay; i >= 0; i--) {
                if (timeSeries[i] <= eighthDeaths) {
                    eighthDay = i;
                    break;
                }
            }

            nowToHalf = nowDay - halfDay;
            halfToQuarter = halfDay - quarterDay;
            quarterToEighth = quarterDay - eighthDay;

            Integer[] doublingDaySpan = {quarterToEighth, halfToQuarter, nowToHalf};
            globalDaysToDouble.put(country, doublingDaySpan);
        }

        return globalDaysToDouble;
    }

    private static void q1PrintIntegerArrayFromMap(FileWriter writer, Map<String, Integer[]> globalTimeSeries, List<String> sortedListCountries, Boolean printCountryLabel) throws IOException {
        for (String country : sortedListCountries) {
            if (printCountryLabel) {
                writer.write(country + ": ");
            }

            Integer[] timeSeries = globalTimeSeries.get(country);
            for (int i = 0; i < timeSeries.length; i++) {
                writer.write(String.valueOf(timeSeries[i]));
                if (i < timeSeries.length - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n");
        }
    }

    private static Map<String, Integer[]> q2CalcDiffTimeSeries(Map<String, Integer[]> countryTimeSeries) {
        Map<String, Integer[]> globalDiffTimeSeries = new HashMap<>();

        for (String country : countryTimeSeries.keySet()) {
            Integer[] timeSeries = countryTimeSeries.get(country);
            Integer[] diffTimeSeries = new Integer[timeSeries.length - 1];
            for (int i = 0; i < timeSeries.length - 1; i++) {
                diffTimeSeries[i] = timeSeries[i + 1] - timeSeries[i];
            }

            globalDiffTimeSeries.put(country, diffTimeSeries);
        }

        return globalDiffTimeSeries;
    }

    /*
     * Sum deaths per country, where csv is sometimes per state/region.
     */
    public static Map<String, Integer[]> q1SumCountryTimeSeries(String file) {

        Map<String, Integer> numOfOccurrences = new HashMap<>();
        Map<String, Integer[]> globalData = new HashMap<>();

        try {
            Scanner sc = new Scanner(new File(file));
            String firstLineColumnNames = sc.nextLine();
            int totalNumColumns = firstLineColumnNames.split(",").length;

            while (sc.hasNext()) {
                String[] line = sc.nextLine().split(",");

                Integer[] countryTimeSeries = new Integer[totalNumColumns - 4]; // ignore first four columns - state, country, lat, long
                for (int i = 0; i < countryTimeSeries.length; i++) {
                    countryTimeSeries[i] = Integer.valueOf(line[i + 4]);
                }

                if (globalData.containsKey(line[1])) {
                    // get previous values
                    Integer[] prevData = globalData.get(line[1]);

                    // sum values for the same country
                    for (int j = 0; j < prevData.length; j++) {
                        countryTimeSeries[j] = countryTimeSeries[j] + prevData[j];
                    }
                    numOfOccurrences.put(line[1], numOfOccurrences.get(line[1]) + 1); // increase counter
                    globalData.put(line[1], countryTimeSeries);

                } else {
                    globalData.put(line[1], countryTimeSeries);
                    numOfOccurrences.put(line[1], 1);
                }
            }
            sc.close();
        } catch (FileNotFoundException e) {
            System.out.println("The input file cannot be found! ");
        }

        return globalData;
    }

    /*
     * Before using the input csv file, I deleted commas for two cells:
     * "Korea South" and "Bonaire Sint Eustatius and Saba".
     * You might need to do that as well if you decide to use this code for
     * preprocessing.
     */
    public static Map<String, Double[]> preprocessData(String file) {

        Map<String, Integer> numOfOccurrences = new HashMap<String, Integer>();
        Map<String, Double[]> globalData = new HashMap<String, Double[]>();

        try {
            Scanner sc = new Scanner(new File(file));
            String firstLineColumnNames = sc.nextLine();
            int totalNumColumns = firstLineColumnNames.split(",").length;

            while (sc.hasNext()) {
                String[] line = sc.nextLine().split(",");

                Double[] countryInfo = new Double[3];
                countryInfo[0] = Double.valueOf(line[2]); // latitude
                countryInfo[1] = Double.valueOf(line[3]); // longitude
                countryInfo[2] = Double.valueOf(line[totalNumColumns - 1]); // total number of deaths on the last day, i.e.last column

                if (globalData.containsKey(line[1])) {
                    // get previous values
                    Double[] prevData = globalData.get(line[1]);

                    // sum values for the same country
                    for (int j = 0; j < prevData.length; j++) {
                        countryInfo[j] = countryInfo[j] + prevData[j];
                    }
                    numOfOccurrences.put(line[1], numOfOccurrences.get(line[1]) + 1); // increase counter
                    globalData.put(line[1], countryInfo);

                } else {
                    globalData.put(line[1], countryInfo);
                    numOfOccurrences.put(line[1], 1);
                }
            }

            // find average for latitude and longitude for countries that have several occurrences
            for (String country : numOfOccurrences.keySet()) {

                int count = numOfOccurrences.get(country);
                if (count > 1) {
                    Double[] list = globalData.get(country);

                    // index 0 - latitude
                    // index 1 - longitude
                    list[0] = (double) list[0] / count;
                    list[1] = (double) list[1] / count;

                    globalData.put(country, list);
                }
            }
            sc.close();
        } catch (FileNotFoundException e) {
            System.out.println("The input file cannot be found! ");
        }

        return globalData;
    }

    /*
     * This method computes single-linkage distance between two clusters.
     * Makes use of Manhattan distance.
     */
    public static double singleLinkageDistance(Map<String, Double[]> globalData, List<String> cluster1, List<String> cluster2) {

        double minDistance = Double.MAX_VALUE;
        double distance;
        for (int i = 0; i < cluster1.size(); i++) {
            for (int j = 0; j < cluster2.size(); j++) {

                distance = 0;

                Double[] values1 = globalData.get(cluster1.get(i));
                Double[] values2 = globalData.get(cluster2.get(j));

                for (int m = 0; m < values1.length; m++) {
                    // TODO: change this part of code to use Euclidean distance
                    distance += Math.abs((values1[m] - values2[m]));
                }
                if (distance < minDistance) minDistance = distance;
            }
        }
        return minDistance;
    }

    /*
     * This method given input on parameters for each country, and information about
     * means for the clusters, assigns each country to the cluster and updates the
     * array holding information about the assignment.
     */
    private static void findClusterForCountries(List<Double[]> data, List<Double[]> means, int[] cluster_info) {

        for (int i = 0; i < data.size(); i++) {
            Double[] countryInfo = data.get(i);

            int min_cluster = 0;
            double min_distance = Double.MAX_VALUE;

            for (int k = 0; k < K; k++) {

                Double[] mean_vector = means.get(k);
                double distance = 0;

                // Manhattan distance
                for (int j = 0; j < countryInfo.length; j++) {
                    // TODO: change this to Euclidean distance instead of Manhattan
                    distance += Math.abs(countryInfo[j] - mean_vector[j]);
                }

                if (distance < min_distance) {
                    min_cluster = k + 1;
                    min_distance = distance;
                }
            }
            cluster_info[i] = min_cluster;
        }
    }

    private static Map<String, Integer> q7KMeans(FileWriter writer, List<String> sortedCountryStringList, List<Double[]> global_data) throws IOException {
        Map<String, Integer> clusterData = new HashMap<>();

        int[] cluster_info = new int[sortedCountryStringList.size()]; // cluster to which country is assigned to

        // K vectors of length FEATURE_LENGTH, those will hold our cluster centers
        List<Double[]> means = new ArrayList<>();
        // Choose K random countries as our initial means for cluster centers
        for (int i = 0; i < K; i++) {
            Random r = new Random();
            int rand = r.nextInt(sortedCountryStringList.size());
            Double[] vector = global_data.get(rand);
            means.add(vector);
        }

        // find initial cluster assignment for all countries
        q7FindClusterForCountriesEuclidean(global_data, means, cluster_info);

        // start updating means for clusters until clusters do not change
        boolean keepUpdatingMeans = true;

        List<Double[]> previous_iteration_means = means;

        while (keepUpdatingMeans) {

            List<Double[]> recomputed_means = recomputeMeans(global_data, cluster_info);

            // need to make sure that all K means are not changing for stopping learning algorithm
            int checker = 0;
            for (int cl_index = 0; cl_index < K; cl_index++) {

                if (Arrays.equals(recomputed_means.get(cl_index), previous_iteration_means.get(cl_index))) {
                    checker += 1;
                }
            }
            if (checker == K) {
                // all K vectors are still the same after last iteration of updates to the means
                // therefore stop the algorithm
                keepUpdatingMeans = false;
                finalMeans = recomputed_means;
                break;
            }
            q7FindClusterForCountriesEuclidean(global_data, recomputed_means, cluster_info);
            previous_iteration_means = recomputed_means;
        }

        // Print out list of countries in each cluster for K-Means
        System.out.println("K-Means Clustering Results: ");
        for (int i = 1; i < K + 1; i++) {
            System.out.print(i - 1 + " [");
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < sortedCountryStringList.size(); j++) {
                if (cluster_info[j] == i) {
                    sb.append(sortedCountryStringList.get(j) + ", ");
                }
            }
            if (sb.length() != 0) System.out.print(sb.toString().substring(0, sb.length() - 2));
            else System.out.print(sb.toString());
            System.out.print("]");
            System.out.println();
        }

        for (int i = 0; i < cluster_info.length; i++) {
            writer.write(sortedCountryStringList.get(i) + ": " + String.valueOf(cluster_info[i] - 1));
            writer.write("\n");
        }

        for (int i = 0; i < sortedCountryStringList.size(); i++) {
            String country = sortedCountryStringList.get(i);
            int clusterIndex = cluster_info[i] - 1;
            clusterData.put(country,clusterIndex);
        }

        return clusterData;
    }

    private static void q7FindClusterForCountriesEuclidean(List<Double[]> data, List<Double[]> means, int[] cluster_info) {

        for (int i = 0; i < data.size(); i++) {
            Double[] countryInfo = data.get(i);

            int min_cluster = 0;
            double min_distance = Double.MAX_VALUE;

            for (int k = 0; k < K; k++) {

                Double[] mean_vector = means.get(k);

                double x1 = countryInfo[0];
                double y1 = countryInfo[1];
                double z1 = countryInfo[2];

                double x2 = mean_vector[0];
                double y2 = mean_vector[1];
                double z2 = mean_vector[2];

                double distance = q5CalcEuclidean(x1, y1, z1, x2, y2, z2);

                if (distance < min_distance) {
                    min_cluster = k + 1;
                    min_distance = distance;
                }
            }
            cluster_info[i] = min_cluster;
        }
    }

    /*
     * This method given all countries' information and a new assignment of countries to the clusters,
     * recomputes the means for clusters by finding the average.
     */
    private static List<Double[]> recomputeMeans(List<Double[]> countries, int[] cluster_info) {

        List<Double[]> means = new ArrayList<Double[]>();

        // consider each cluster
        for (int i = 0; i < K; i++) {

            int cluster_size = 0;
            Double[] cluster_mean = new Double[FEATURE_LENGTH];

            // initialize values to 0, to avoid null pointer
            for (int k = 0; k < cluster_mean.length; k++) {
                cluster_mean[k] = 0.0;
            }

            for (int j = 0; j < cluster_info.length; j++) {

                if (cluster_info[j] == i + 1) {

                    cluster_size += 1;

                    Double[] countryInfo = countries.get(j);

                    // sum values for all countries in the cluster
                    for (int index = 0; index < countryInfo.length; index++) {
                        cluster_mean[index] += countryInfo[index];
                    }
                }
            }

            // average all countries in the cluster
            for (int index = 0; index < cluster_mean.length; index++) {
                cluster_mean[index] = cluster_mean[index] / (double) cluster_size;
            }
            means.add(cluster_mean);
        }
        return means;
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