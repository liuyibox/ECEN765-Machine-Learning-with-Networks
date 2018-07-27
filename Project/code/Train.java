public class Train {

	public int numLayers;
	private float alpha = 0.005f; 		// Our training rate.
	
	private float [] input;				// Training input.
	private float [] output;			// Expected output.
	
	private float [] annOutput;			// Output layers actual output.
	private float [] hiddenOutput;		// Hidden layers actual output.
	
	private float[] outputDelta;		// Output layers calc error. 
	private float[] hiddenDelta;		// Hidden layers calc error.
	
	public float [][] w12;				// inputLayer(1) -> hiddenLayers(2) weights.
	public float [][] w23;				// hiddenLayers(2) -> outputLayers(3) weights.
	
	final int OUTPUT_SIZE = 4;
	final int HIDDEN_SIZE = 180;
	final int INPUT_SIZE = 360;
	
	public Train(NeuralNetwork nn, float[] in, float[] out) {
		this.numLayers = nn.numLayers;	
		input = in;		
		output = out;							// Expected output.
		annOutput = nn.output;					// Actual output.
		outputDelta = new float[OUTPUT_SIZE];
		w12 = nn.weights1;
		if(numLayers == 2) {					// If two layer, set these values.
			hiddenOutput = nn.hiddenOutput;
			hiddenDelta = new float[HIDDEN_SIZE];
			w23 = nn.weights2;
		}
	}
	
	// Calculate error.
	public void backprop() {
		if(numLayers == 1) {																	   // 1-layer case error calc.
			for(int j = 0; j < OUTPUT_SIZE; j++) {							
				outputDelta[j] = (annOutput[j]*(1-annOutput[j]))*(output[j]-annOutput[j]);	
			}	
		}
		else {																						// 2-layer error calc.
			for(int k = 0; k < OUTPUT_SIZE; k++) {													// Output error: For each output value,
				outputDelta[k] = (annOutput[k]*(1-annOutput[k]))*(output[k]-annOutput[k]);			// Set output neurons error. 
			}																						// Hidden error:
			
			for(int i = 0; i < HIDDEN_SIZE; i++) {												// Update Network weights from hidden->Output.
				for(int j = 0; j < OUTPUT_SIZE; j++) {
					w23[i][j] = w23[i][j] + alpha*outputDelta[j]*hiddenOutput[i];
				}
			}
			
			for(int j = 0; j < HIDDEN_SIZE; j++) { 												    // For each hidden neuron,
				float temp = 0;
				for(int k = 0; k < OUTPUT_SIZE; k++) {												// Get the sum of its weights.
					temp += w23[j][k]*outputDelta[k];												
				}
				hiddenDelta[j] = (hiddenOutput[j]*(1-hiddenOutput[j]))*temp;						// Calculate the hidden error.
			}
			
			for(int i = 0; i < INPUT_SIZE; i++) {												// Update Network weights from input->hidden.
				for(int j = 0; j < HIDDEN_SIZE; j++) {
					w12[i][j] = w12[i][j] + alpha*hiddenDelta[j]*input[i];
				}
			}
		}
//		updateWeights();
	}
	
	// Update weights based on calculated error.
	void updateWeights() {
		if(numLayers == 1) {																    // 1-layer case.
			for(int j = 0; j < OUTPUT_SIZE; j++) {												// Update input->output weights.
				for(int i = 0; i < INPUT_SIZE; i++) {
					w12[i][j] = w12[i][j] + alpha*input[i]*outputDelta[j];	
				}
			}
		}
		else {																					// 2-layer case.
			for(int i = 0; i < INPUT_SIZE; i++) {												// Update Network weights from input->hidden.
				for(int j = 0; j < HIDDEN_SIZE; j++) {
					w12[i][j] = w12[i][j] + alpha*hiddenDelta[j]*input[i];
				}
			}
			for(int i = 0; i < HIDDEN_SIZE; i++) {												// Update Network weights from hidden->Output.
				for(int j = 0; j < OUTPUT_SIZE; j++) {
					w23[i][j] = w23[i][j] + alpha*outputDelta[j]*hiddenOutput[i];
				}
			}
		}
	}
	
}
