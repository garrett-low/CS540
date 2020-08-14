import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

/*
 * Author: Ainur Ainabekova, TA
 *
 * This is a solution of the programming assignment 5 for CS-540 Summer 2020.
 * This program solves maze using BFS and A* Search algorithms.
 *
 * When parsing SVG file this program assumes that the start of the maze is at the top
 * and finish is at the bottom.
 *
 */

/*
	TODO: Implement DFS Algorithm 
	TODO: Implement method for printing the maze and maze with solution in the required format
	TODO: Implement A* Search with Euclidean distance (current version is using Manhattan distance)
 */

public class MazeSolver {
  
  // make sure to change these values based on the maze dimensions
  // and modify svg file path
  private static final int WIDTH = 57;
  private static final int HEIGHT = 58;
  private static final String SVGFILEPATH = "./grlow_maze.svg";
  
  private static final DateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd-HH.mm.ss.SSS");
  private static final String strDate = dateFormat.format(Calendar.getInstance().getTime());
  private static final String q1PlotOutputPath = "./temp/q1-plot-" + strDate + ".txt";
  private static final String q2SuccOutputPath = "./temp/q2-succ-" + strDate + ".txt";
  private static final String q3SolutionOutputPath = "./temp/q3-solution-" + strDate + ".txt";
  private static final String q4PlotSolutionOutputPath = "./temp/q4-plot_solution-" + strDate + ".txt";
  
  public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
    createFile(q1PlotOutputPath);
    
    SVGParser svgParser = new SVGParser(SVGFILEPATH, WIDTH, HEIGHT);
    
    svgParser.parseXML();
    
    Cell[] startAndFinishCells = svgParser.generateMaze();
    
    try (FileWriter writer = new FileWriter(q1PlotOutputPath)) {
      svgParser.printMaze(writer);
    }
    
    // identify start and finish cells
    Cell start = startAndFinishCells[0];
    Cell finish = startAndFinishCells[1];
    
    System.out.println("\nPrinting Successor Matrix:");
    svgParser.printSuccessorMatrix();
    try (FileWriter writer = new FileWriter(q2SuccOutputPath)) {
      svgParser.printToFileSuccessorMatrix(writer);
    }
    
    System.out.println("\nStarted running BFS Algorithm !");
    
    BFS(start, finish);
    
    System.out.println("\nStarted running A* Search Algorithm !");
    
    // call A* search on this maze
    aStarSearch(start, finish);
    
    System.out.print("\nPrinting the solution path:");
    ArrayList<Integer[]> solutionPath = printSolution(finish);
    try (FileWriter writer = new FileWriter(q3SolutionOutputPath)) { // Q4
      printToFileSolution(finish, writer);
    }
    try (FileWriter writer = new FileWriter(q4PlotSolutionOutputPath)) {
      svgParser.printMazeWithSolution(writer, solutionPath);
    }
  }
  
  /*
   * BFS Algorithm
   * This method returns the number of visited cells.
   */
  public static int BFS(Cell start, Cell finish) {
    
    HashSet<Cell> visited = new HashSet<Cell>();
    LinkedList<Cell> queue = new LinkedList<Cell>();
    boolean reachedFinish = false;
    
    queue.add(start);
    visited.add(start);
    
    while (!queue.isEmpty() && !reachedFinish) {
      Cell curr = queue.poll();
      if (curr.xCoord == finish.xCoord && curr.yCoord == finish.yCoord) {
        reachedFinish = true;
      }
      
      ArrayList<Cell> neighbors = curr.getNeighbors();
      for (Cell neighbor : neighbors) {
        if (!visited.contains(neighbor)) {
          visited.add(neighbor);
          queue.add(neighbor);
        }
      }
    }
    System.out.println("Number of Expanded Vertices = " + visited.size());
    
    return visited.size();
  }
  
  /*
   *
   * This method represents A* search algorithm.
   * It returns the number of total expanded cells for reaching finish cell from start cell.
   *
   */
  public static int aStarSearch(Cell start, Cell finish) {
    
    HashSet<Cell> visited = new HashSet<Cell>();
    boolean reachedFinish = false;
    
    // priority queue based on the f value
    PriorityQueue<Cell> queue = new PriorityQueue<Cell>(WIDTH * HEIGHT, new Comparator<Cell>() {
      
      public int compare(Cell cell1, Cell cell2) {
        if (cell1.f > cell2.f) {
          return 1;
        }
        if (cell1.f < cell2.f) {
          return -1;
        }
        return 0;
      }
      
    });
    
    // initialize g cost for the start cell
    start.g = 0;
    queue.add(start);
    
    while (!queue.isEmpty() && !reachedFinish) {
      
      // retrieve cell with the lowest f value
      Cell current = queue.poll();
      
      // to track cells that were expanded
      visited.add(current);
      
      // case when finish cell is dequeued
      if (current.xCoord == finish.xCoord && current.yCoord == finish.yCoord) {
        reachedFinish = true;
      }
      
      ArrayList<Cell> neighbors = current.getNeighbors();
      
      // consider all the neighbors of the current cell
      for (Cell adjacentCell : neighbors) {
        
        // since one step is needed to move from current cell to its neighbor, increment g value
        double g = current.g + 1;
        
        // Manhattan distance is used to compute h from adjacent cell to the finish cell
        // f = g + h
        double f = g + Math.abs(adjacentCell.xCoord - finish.xCoord) +
            Math.abs(adjacentCell.yCoord - finish.yCoord);
        
        // if this cell was already expanded then do not add to the queue;
        // for our maze examples we do not need to worry about cases when there are
        // multiple paths to the same cell and we could have several f costs for same cell (no loops)
        if (visited.contains(adjacentCell)) {
          continue;
        } else {
          
          // set f and g values for adjacent cell
          adjacentCell.g = g;
          adjacentCell.f = f;
          
          // need this for backtracking purposes
          adjacentCell.parent = current;
          
          // add adjacent cell to the queue so that it's expanded later
          queue.add(adjacentCell);
        }
      }
    }
    
    System.out.println("Number of Expanded Vertices = " + visited.size());
    
    return visited.size();
    
  }
  
  public static ArrayList<Integer[]> printSolution(Cell finish) {
    
    Cell curr = finish;
    Cell prev = finish.parent;
    
    System.out.println();
    ArrayList<Integer[]> solutionPath = new ArrayList<>();
    Integer[] solutionPlotPoint = {finish.xCoord, finish.yCoord};
    solutionPath.add(solutionPlotPoint);
    
    StringBuilder sb = new StringBuilder();
    
    while (prev != null) {
      if (curr.xCoord < prev.xCoord) {
        sb.append("L");
      }
      if (curr.xCoord > prev.xCoord) {
        sb.append("R");
      }
      if (curr.yCoord > prev.yCoord) {
        sb.append("D");
      }
      if (curr.yCoord < prev.yCoord) {
        sb.append("U");
      }
      curr = prev;
      // add curr to the solution path
      Integer[] currPlotPoint = {curr.xCoord, curr.yCoord};
      solutionPath.add(currPlotPoint);
      
      prev = curr.parent;
    }
    System.out.println(sb.reverse().toString());
    
    // reverse the solutionPath list
    Collections.reverse(solutionPath);
    return solutionPath;
  }
  
  public static void printToFileSolution(Cell finish, FileWriter writer) throws IOException {
    Cell curr = finish;
    Cell prev = finish.parent;
    
    System.out.println();
    
    StringBuilder sb = new StringBuilder();
    
    while (prev != null) {
      if (curr.xCoord < prev.xCoord) {
        sb.append("L");
      }
      if (curr.xCoord > prev.xCoord) {
        sb.append("R");
      }
      if (curr.yCoord > prev.yCoord) {
        sb.append("D");
      }
      if (curr.yCoord < prev.yCoord) {
        sb.append("U");
      }
      curr = prev;
      prev = curr.parent;
    }
    writer.write(sb.reverse().toString());
  }
  
  public static void createFile(String outputPath) {
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
