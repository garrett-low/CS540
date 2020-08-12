import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

public class p4 {

	private static final int K = 8;               // number of clusters to generate
	private static final int FEATURE_LENGTH = 3;  // number of parameters used in the algorithms

	public static void main(String[] args) {

		/*
		 * Latitude, longitude and total number of deaths are used as parameters.
		 *  - for latitude and longitude average value is taken if country appears more than once;
		 *  - for total number of deaths sum of the last column is used if country appears more than once.
		 */
		/*
		 * TODO: For your submission you need to implement Part 1 for finding suitable parameters. 
		 */
		Map<String, Double[]> globalData = preprocessData("./time_series_covid19_deaths_global.csv");

		Set<String> countries = globalData.keySet();
		List<String> sortedListCountries = new ArrayList<>(countries);
		Collections.sort(sortedListCountries);
		
		List<Double[]> global_data = new ArrayList<Double[]>();
		for (String country: sortedListCountries){
			global_data.add(globalData.get(country));
		}

		// Hierarchical Clustering with single-linkage and Manhattan distance
		List<List<String>> clusters = new ArrayList<List<String>>();

		// each country appears as a separate cluster at the beginning
		for (int i=0; i<sortedListCountries.size(); i++){
			List<String> cluster = new ArrayList<String>();
			cluster.add(sortedListCountries.get(i));
			clusters.add(cluster);
		}

		while (clusters.size() != K ){

			double minimumDistance = Double.MAX_VALUE;
			List<String> clusterToMerge1 = null, clusterToMerge2 = null;
			int indexCluster1 = -1, indexCluster2 = -1;

			for (int i=0; i<clusters.size(); i++){
				List<String> cluster1 = clusters.get(i);
				for (int j=i+1; j<clusters.size(); j++){ // make sure it's not the same cluster though, otherwise dist is 0.

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
			if (clusterToMerge1!=null && clusterToMerge2!=null){

				clusters.remove(indexCluster2);
				clusters.remove(indexCluster1);

				clusterToMerge1.addAll(clusterToMerge2);
				clusters.add(clusterToMerge1);
			} else{
				System.out.println("Something is incorrect, one cluster might be null.");
			}
		}

		// Print out final clusters
		System.out.println("Hierarchical Clustering Results: ");
		int cluster_num = 0;
		HashMap<String, Integer> clusterHier = new HashMap<String, Integer>();
		for (List<String> l : clusters){
			System.out.println(cluster_num + " " + l);

			for (int i=0; i<l.size(); i++){
				String country = l.get(i);
				clusterHier.put(country, cluster_num);
			}
			cluster_num++;
		}
		System.out.println("=============================================================");

		int[] cluster_info = new int[sortedListCountries.size()]; // cluster to which country is assigned to

		// K-Means Clustering with Manhattan distance

		// K vectors of length FEATURE_LENGTH, those will hold our cluster centers
		List<Double[]> means = new ArrayList<Double[]>(); 
		// Choose K random countries as our initial means for cluster centers
		for (int i=0; i < K; i++){
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

		while (keepUpdatingMeans){

			List<Double[]> recomputed_means = recomputeMeans(global_data, cluster_info);

			// need to make sure that all K means are not changing for stopping learning algorithm
			int checker = 0;
			for(int cl_index=0; cl_index<K; cl_index++){

				if (Arrays.equals(recomputed_means.get(cl_index), previous_iteration_means.get(cl_index))){
					checker+=1;
				}
			}
			if (checker == K){
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
		for (int i=1; i<K+1; i++){
			System.out.print(i-1 + " [");
			StringBuilder sb = new StringBuilder();
			for (int j=0; j<sortedListCountries.size(); j++){
				if (cluster_info[j] == i){
					sb.append(sortedListCountries.get(j) + ", ");
				}
			}
			if (sb.length() != 0)System.out.print(sb.toString().substring(0,sb.length()-2));
			else System.out.print(sb.toString());
			System.out.print("]");
			System.out.println();
		}

	}

	/*
	 * Before using the input csv file, I deleted commas for two cells: 
	 * "Korea South" and "Bonaire Sint Eustatius and Saba". 
	 * You might need to do that as well if you decide to use this code for
	 * preprocessing. 
	 */
	public static Map<String, Double[]> preprocessData(String file) {

		Map<String, Integer> numOfOccurrences = new HashMap<String,Integer>();
		Map<String,Double[]> globalData = new HashMap<String, Double[]>();

		try {
			Scanner sc = new Scanner(new File(file));
			String firstLineColumnNames = sc.nextLine();
			int totalNumColumns = firstLineColumnNames.split(",").length;

			while (sc.hasNext()){
				String[] line = sc.nextLine().split(",");

				Double[] countryInfo = new Double[3];
				countryInfo[0] = Double.valueOf(line[2]); // latitude
				countryInfo[1] = Double.valueOf(line[3]); // longitude
				countryInfo[2] = Double.valueOf(line[totalNumColumns-1]); // total number of deaths on the last day, i.e.last column

				if (globalData.containsKey(line[1])){
					// get previous values
					Double[] prevData = globalData.get(line[1]);

					// sum values for the same country
					for (int j=0; j<prevData.length; j++){
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
			for (String country: numOfOccurrences.keySet()){

				int count = numOfOccurrences.get(country);
				if (count > 1){
					Double[] list = globalData.get(country);

					// index 0 - latitude
					// index 1 - longitude
					list[0] = (double) list[0]/count;
					list[1] = (double) list[1]/count;

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
	public static double singleLinkageDistance( Map<String, Double[]> globalData, List<String> cluster1, List<String> cluster2){

		double minDistance = Double.MAX_VALUE;
		double distance; 
		for (int i=0; i<cluster1.size(); i++){
			for (int j=0; j<cluster2.size(); j++){

				distance = 0;

				Double[] values1 = globalData.get(cluster1.get(i));
				Double[] values2 = globalData.get(cluster2.get(j));

				for (int m=0; m<values1.length; m++){
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

		for (int i=0; i<data.size(); i++){
			Double[] countryInfo = data.get(i);

			int min_cluster = 0;
			double min_distance = Double.MAX_VALUE;

			for (int k = 0; k<K; k++){

				Double[] mean_vector = means.get(k);
				double distance = 0;

				// Manhattan distance 
				for (int j=0; j < countryInfo.length; j++){
					// TODO: change this to Euclidean distance instead of Manhattan
					distance += Math.abs(countryInfo[j] - mean_vector[j]); 
				}

				if (distance<min_distance){
					min_cluster = k+1;
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
	private static List<Double[]> recomputeMeans(List<Double[]> countries, int[] cluster_info){
		
		List<Double[]> means = new ArrayList<Double[]>();
		
		// consider each cluster
		for (int i=0; i < K; i++){

			int cluster_size = 0;
			Double[] cluster_mean = new Double[FEATURE_LENGTH];
			
			// initialize values to 0, to avoid null pointer
			for (int k=0;k<cluster_mean.length; k++){
				cluster_mean[k] = 0.0;
			}

			for (int j=0; j< cluster_info.length; j++){

				if (cluster_info[j] == i+1){

					cluster_size += 1;

					Double[] countryInfo = countries.get(j);

					// sum values for all countries in the cluster
					for (int index = 0; index<countryInfo.length; index++){
						cluster_mean[index] += countryInfo[index];
					}
				}
			}

			// average all countries in the cluster 
			for (int index = 0; index<cluster_mean.length; index++){
				cluster_mean[index] = cluster_mean[index]/(double)cluster_size;
			}
			means.add(cluster_mean);
		}
		return means;
	}
}
