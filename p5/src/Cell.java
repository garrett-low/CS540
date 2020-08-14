import java.util.ArrayList;

/*
 * This class represents a cell in the maze.
 * 
 */
public class Cell {
	
	public int xCoord;
	public int yCoord;
	public ArrayList<Cell> neighbors;
	public double g;       // for A* search
	public double f;       // for A* search
	public Cell parent;    // for backtracking
	
	public Cell(int xCoord, int yCoord){
		this.xCoord = xCoord;
		this.yCoord = yCoord;
	}

	public ArrayList<Cell> getNeighbors() {
		return neighbors;
	}

	public void setNeighbors(ArrayList<Cell> neighbors) {
		this.neighbors = neighbors;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Cell) {
			Cell cell = (Cell) obj;
			return (this.xCoord == cell.xCoord) && (this.yCoord == cell.yCoord);
		} else {
			return  false;
		}
	}
	
	@Override
	public int hashCode() {
		int result = 17;
		result = 31 * result + this.xCoord;
		result = 31 * result + this.yCoord;
		return result;
	}

}
