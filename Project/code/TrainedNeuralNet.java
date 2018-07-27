import java.io.IOException;

class TrainedNeuralNet {
	
	private static float[][][] inData;                                               
    private static float[][] outData;
    private static int[][] labelData;
    
    private static float[][][] nnOutputData;
	int count;
    
    // Generates training data.
	public static void trainingData() {                                         
//        ReadFile readData = new ReadFile();				
//        inData = readData.readStuff("training2.txt");   // Get training data. 26*35
		ReadWriteExcelFile rw = new ReadWriteExcelFile();
		try {
			rw.readXLSFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		inData = rw.rawData;							//20*10*360
        outData = new float[4][4];											     // Get expected outputs.
        for(int i = 0; i < 4; ++i) {
            outData[i][i] = 1f;                                          
        }
        labelData = rw.labelData;						//20*10
    }
	
	// Train our neural network.
  	public NeuralNetwork getTrainedNn() {
  		nnOutputData = new float[20][10][4];
  		
    	trainingData();		
    	int numLayers = 2;
    	NeuralNetwork nn = new NeuralNetwork(numLayers);
/*    	for(int i = 0; i < 700; ++i) {
    		for(int j = 0; j < 26; j++) {								// For each training example,  			
    			nn.fordprop(inData[j]);									// Calculate weighted sums & activation/output values.
		        Train train = new Train(nn, inData[j], outData[j]);
				train.backprop();	      								// Train weights.
    		}
        }      
*/		
    	for(int iter = 0; iter < 500; iter++){
	    	int a = 3;
	    	
    		for(int i = 0; i < inData.length; i++){
    			int b = inData.length;
				for(int j = 0; j < inData[0].length; j++){
					int c = inData[0].length;
				//	int index = i*10+j;
					if(labelData[i][j] == 0) continue;
					nn.fordprop(inData[i][j]);
					nnOutputData[i][j] = nn.output;
					Train train = new Train(nn, inData[i][j], outData[labelData[i][j]-1]);
					train.backprop();
					nn.weights1 = train.w12;
					nn.weights2 = train.w23;
				}
			}
	    	
    		count = 0;
    		for(int i = 0; i < inData.length; i++){
				for(int j = 0; j < inData[0].length; j++){
					if(labelData[i][j] == 0) continue;
			        if(isMax(nnOutputData[i][j],labelData[i][j]-1)) 
			        	count++;
				}
    		}
    	}
    	return nn;
    }
  	
	private static boolean isMax(float[] a, int i){
		for(int j = 0; j < a.length; j++){
			if(a[j] > a[i]) return false;										// a[j] other values is greater than a[i] <-- our value train again
		}
		return true;
	}
  	
  	public static void main(String[] args) {
    	TrainedNeuralNet main = new TrainedNeuralNet();
    	NeuralNetwork nn = main.getTrainedNn();
    }
}
    		
