import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import org.apache.poi.hssf.usermodel.HSSFCell;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;

import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class ReadWriteExcelFile {

	public static float rawData[][][];
	public static int labelData[][];
	
	public static void writeTextFile(String filename, float[][] matrix) throws IOException{
		
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(filename, true));
			int len = matrix.length;
			for(int i = 0; i < matrix.length; i++){
				for(int j = 0; j < matrix[i].length; j++){
					bw.write(matrix[i][j] + " " );
				}
				bw.newLine();
			}
			bw.flush();
		}catch(IOException e){}
	}
	
	public static void writeLabelFile(String filename, int[][] matrix) throws IOException{
		
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(filename, true));
			int len = matrix.length;
			for(int i = 0; i < matrix.length; i++){
				for(int j = 0; j < matrix[i].length; j++){
					bw.write(matrix[i][j]+"");
					bw.newLine();
				}
			}
			bw.flush();
		}catch(IOException e){}
	}
	
	public static void readXLSFile() throws IOException	{
		float currentData[][] = new float[10][360]; 
		float res[][][] = new float[20][10][360];
		int label[][] = new int[20][10];
		
		InputStream ExcelFileToRead = new FileInputStream("log_prob_plot.xls");
		HSSFWorkbook wb = new HSSFWorkbook(ExcelFileToRead);

		String sheetName = "";
		for(int s = 0; s < 20; s++){
			sheetName = s*10+1 + "-" + (s+1)*10;
			
			HSSFSheet sheet=wb.getSheet(sheetName);
			HSSFRow row; 
			HSSFCell cell;
			
			HSSFCell cell1;
			HSSFCell cell2;
			HSSFCell cell3;
			HSSFCell cell4;
			HSSFCell cell5;
			HSSFCell cell6;
			HSSFCell cell7;
			HSSFCell cell8;
			HSSFCell cell9;
			HSSFCell cell10;

			Iterator rows = sheet.rowIterator();
			
			if(rows.hasNext()){
				row=(HSSFRow) rows.next();
				
				cell1 = row.getCell(2);
				int label1 = (int) cell1.getNumericCellValue();
				label[s][0] = label1;
				
				cell2 = row.getCell(6);
				int label2 = (int) cell2.getNumericCellValue();
				label[s][1] = label2;
				
				cell3 = row.getCell(10);
				int label3 = (int) cell3.getNumericCellValue();
				label[s][2] = label3;
				
				cell4 = row.getCell(14);
				int label4 = (int) cell4.getNumericCellValue();
				label[s][3] = label4;
				
				cell5 = row.getCell(18);
				int label5 = (int) cell5.getNumericCellValue();
				label[s][4] = label5;
				
				cell6 = row.getCell(22);
				int label6 = (int) cell6.getNumericCellValue();
				label[s][5] = label6;
				
				cell7 = row.getCell(26);
				label[s][6] = (int) cell7.getNumericCellValue();
				
				cell8 = row.getCell(30);
				label[s][7] = (int) cell8.getNumericCellValue();
				
				cell9 = row.getCell(34);
				label[s][8] = (int) cell9.getNumericCellValue();
				
				cell10 = row.getCell(38);
				label[s][9] = (int) cell10.getNumericCellValue();
			}
			
			int rowNumber = 0;
			while (rows.hasNext())
			{
				row=(HSSFRow) rows.next();
				
				cell1 = row.getCell(1);
				float cellValue1 = (float) cell1.getNumericCellValue();
				currentData[0][rowNumber] = cellValue1;
				
				cell2 = row.getCell(5);
				float cellValue2 = (float) cell2.getNumericCellValue();
				currentData[1][rowNumber] = cellValue2;
				
				cell3 = row.getCell(9);
				float cellValue3 = (float) cell3.getNumericCellValue();
				currentData[2][rowNumber] = cellValue3;
				
				cell4 = row.getCell(13);
				float cellValue4 = (float) cell4.getNumericCellValue();
				currentData[3][rowNumber] = cellValue4;
				
				cell5 = row.getCell(17);
				float cellValue5 = (float) cell5.getNumericCellValue();
				currentData[4][rowNumber] = cellValue5;
				
				cell6 = row.getCell(21);
				float cellValue6 = (float) cell6.getNumericCellValue();
				currentData[5][rowNumber] = cellValue6;
				
				cell7 = row.getCell(25);
				float cellValue7 = (float) cell7.getNumericCellValue();
				currentData[6][rowNumber] = cellValue7;
				
				cell8 = row.getCell(29);
				float cellValue8 = (float) cell8.getNumericCellValue();
				currentData[7][rowNumber] = cellValue8;
				
				cell9 = row.getCell(33);
				float cellValue9 = (float) cell9.getNumericCellValue();
				currentData[8][rowNumber] = cellValue9;
				
				cell10 = row.getCell(37);
				float cellValue10 = (float) cell10.getNumericCellValue();
				currentData[9][rowNumber] = cellValue10;
				
				rowNumber++;
			}
			writeTextFile("production_rate_data.txt",currentData);
			
			res[s] = currentData;
		}
		writeLabelFile("label.txt", label);
		rawData = res;
		labelData = label;
//		return res;
	}
	
	public static void writeXLSFile() throws IOException {
		
		String excelFileName = "C:/Test.xls";//name of excel file

		String sheetName = "Sheet1";//name of sheet

		HSSFWorkbook wb = new HSSFWorkbook();
		HSSFSheet sheet = wb.createSheet(sheetName) ;

		//iterating r number of rows
		for (int r=0;r < 5; r++ )
		{
			HSSFRow row = sheet.createRow(r);
	
			//iterating c number of columns
			for (int c=0;c < 5; c++ )
			{
				HSSFCell cell = row.createCell(c);
				
				cell.setCellValue("Cell "+r+" "+c);
			}
		}
		
		FileOutputStream fileOut = new FileOutputStream(excelFileName);
		
		//write this workbook to an Outputstream.
		wb.write(fileOut);
		fileOut.flush();
		fileOut.close();
	}
	
	public static void readXLSXFile() throws IOException
	{
		InputStream ExcelFileToRead = new FileInputStream("4212131673.xls");
		HSSFWorkbook  wb = new HSSFWorkbook(ExcelFileToRead);
		
		XSSFWorkbook test = new XSSFWorkbook(); 
		
		HSSFSheet sheet = wb.getSheet("2-10");
		XSSFRow row; 
		XSSFCell cell;

		Iterator rows = sheet.rowIterator();

		while (rows.hasNext())
		{
			row=(XSSFRow) rows.next();
			Iterator cells = row.cellIterator();
			while (cells.hasNext())
			{
				cell=(XSSFCell) cells.next();
		
				if (cell.getCellType() == XSSFCell.CELL_TYPE_STRING)
				{
					System.out.print(cell.getStringCellValue()+" ");
				}
				else if(cell.getCellType() == XSSFCell.CELL_TYPE_NUMERIC)
				{
					System.out.print(cell.getNumericCellValue()+" ");
				}
				else
				{
					//U Can Handel Boolean, Formula, Errors
				}
			}
			System.out.println();
		}
	
	}
	
	public static void writeXLSXFile() throws IOException {
		
		String excelFileName = "log_prob_plot.xlsx";//name of excel file

		String sheetName = "2-10";//name of sheet

		XSSFWorkbook wb = new XSSFWorkbook();
		XSSFSheet sheet = wb.createSheet(sheetName) ;

		//iterating r number of rows
		for (int r=0;r < 5; r++ )
		{
			XSSFRow row = sheet.createRow(r);

			//iterating c number of columns
			for (int c=0;c < 5; c++ )
			{
				XSSFCell cell = row.createCell(c);
	
				cell.setCellValue("Cell "+r+" "+c);
			}
		}

		FileOutputStream fileOut = new FileOutputStream(excelFileName);

		//write this workbook to an Outputstream.
		wb.write(fileOut);
		fileOut.flush();
		fileOut.close();
	}

//	public static void main(String[] args) throws IOException {
		
//		writeXLSFile();
		
//		String sheet1 = "2-10";
//		readXLSFile(sheet1);

//		readXLSFile();
//		int a = 3;
//		writeXLSXFile();
//		readXLSXFile();

//	}

}
